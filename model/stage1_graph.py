from __future__ import annotations

import json
import math
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from data.embodiment_tags import EMBODIMENT_TAG_MAPPING
from model.kinematic_graph import KinematicGraph, load_graphs
from model.modules.blocks import VectorQuantizer
from model.primitive_tying import frames_to_tokens
from model.stage1 import MotifTokenizer

# Instruction -> id for the table text encoder (same order as the trainers).
TASK_IDS = {
    "Push the cube across the surface to the target location.": 0,
    "Pick up the sphere and place it precisely into the target.": 1,
    "Pull the cube toward the target marker.": 2,
    "Lift the peg and stand it upright.": 3,
    "Pick up the cube.": 4,
    "Stack the cube on top of the other cube.": 5,
}
DEFAULT_ROBOTS = {"panda": "panda_rel", "xarm6": "xarm6_rel", "widowxai": "widowxai_rel"}


# --------------------------------------------------------------------------- text
class TaskTextEmbedder(nn.Module):
    """e_task = L(instruction) from a frozen encoder, then a trainable projection."""

    def __init__(self, out_dim: int, text_encoder: str = "t5-base"):
        super().__init__()
        self.text_encoder = text_encoder
        if text_encoder == "table":
            self.table = nn.Embedding(len(TASK_IDS) + 1, out_dim)
            self.proj = None
        else:
            self.proj = nn.Linear(768, out_dim)
            self._lm = None                       # [model, tokenizer], lazily loaded
            self._cache: Dict[str, torch.Tensor] = {}

    def _encode(self, sentences: List[str], device) -> torch.Tensor:
        missing = [s for s in dict.fromkeys(sentences) if s not in self._cache]
        if missing:
            if self._lm is None:
                from transformers import T5EncoderModel, T5Tokenizer
                lm = T5EncoderModel.from_pretrained(self.text_encoder).eval()
                for p in lm.parameters():
                    p.requires_grad = False
                self._lm = [lm, T5Tokenizer.from_pretrained(self.text_encoder)]
            lm, tok = self._lm
            lm.to(device)
            with torch.no_grad():
                enc = tok(missing, return_tensors="pt", padding=True).to(device)
                hid = lm(**enc).last_hidden_state                       # [n, T, 768]
                mask = enc["attention_mask"].unsqueeze(-1).float()
                pooled = (hid * mask).sum(1) / mask.sum(1).clamp_min(1.0)
            for s, v in zip(missing, pooled):
                self._cache[s] = v.detach()
        return torch.stack([self._cache[s].to(device) for s in sentences])

    def forward(self, sentences: List[str], device) -> torch.Tensor:
        if self.text_encoder == "table":
            ids = torch.tensor([TASK_IDS.get(s, len(TASK_IDS)) for s in sentences], device=device)
            return self.table(ids)
        return self.proj(self._encode(sentences, device))


# --------------------------------------------------------------------------- GNN
class MPNNLayer(nn.Module):
    def __init__(self, hidden: int, d_edge: int, d_task: int, layer_norm: bool = True):
        super().__init__()
        self.w_self = nn.Linear(hidden, hidden)
        self.msg = nn.Sequential(nn.Linear(2 * hidden + d_edge + d_task, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden))
        self.norm = nn.LayerNorm(hidden) if layer_norm else nn.Identity()

    def forward(self, n, src, dst, e, e_task_nodes):
        # message j -> i along edge (src=j, dst=i); n [N, H] (static) or [B, N, H] (per sample), e / e_task_nodes alike
        m = self.msg(torch.cat([n[..., dst, :], n[..., src, :], e, e_task_nodes[..., dst, :]], dim=-1))
        agg = torch.zeros(n.shape, dtype=m.dtype, device=n.device).index_add_(n.dim() - 2, dst, m)   # m may be bf16 under autocast
        return self.norm(F.relu(self.w_self(n) + agg))


class TemporalEncoder(nn.Module):
    """Joint trajectories [..., C, T] -> [..., out]: two strided 1-D convs, pooled to `steps` bins, flattened."""

    def __init__(self, in_ch: int, out: int, hidden: int = 32, steps: int = 8):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv1d(in_ch, hidden // 2, 5, stride=2, padding=2), nn.ReLU(),
                                  nn.Conv1d(hidden // 2, hidden, 5, stride=2, padding=2), nn.ReLU())
        self.steps = steps
        self.out = nn.Linear(hidden * steps, out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lead = x.shape[:-2]
        y = self.conv(x.reshape(-1, *x.shape[-2:]))
        y = F.adaptive_avg_pool1d(y, self.steps).flatten(1)
        return self.out(y).view(*lead, -1)


class TaskConditionedMPNN(nn.Module):
    """L message-passing layers over the disjoint union of the robot graphs, mean-pooled per graph -> g_r.
    Nodes also encode their joint's trajectory over the window (limit-normalised position + per-step change)
    and edges the trajectories of both joints; with trajectories the forward is per sample."""

    DELTA_SCALE = 10.0      # per-step change of the normalised joint position, scaled to O(0.1-1)

    def __init__(self, graphs: List[KinematicGraph], d_task: int, hidden: int = 128,
                 layers: int = 3, layer_norm: bool = True, temporal_dim: int = 32,
                 qpos_layout: List[dict] = None):
        super().__init__()
        x, src, dst, e, gid = [], [], [], [], []
        offset = 0
        for g_i, g in enumerate(graphs):
            x.append(torch.tensor(g.node_feats, dtype=torch.float32))
            src.append(torch.tensor(g.edge_src) + offset)
            dst.append(torch.tensor(g.edge_dst) + offset)
            e.append(torch.tensor(g.edge_feats, dtype=torch.float32))
            gid.append(torch.full((g.num_nodes,), g_i, dtype=torch.long))
            offset += g.num_nodes
        self.register_buffer("x", torch.cat(x), persistent=False)
        self.register_buffer("src", torch.cat(src), persistent=False)
        self.register_buffer("dst", torch.cat(dst), persistent=False)
        self.register_buffer("e", torch.cat(e), persistent=False)
        self.register_buffer("gid", torch.cat(gid), persistent=False)
        self.num_graphs = len(graphs)
        d_node, d_edge = self.x.shape[1], self.e.shape[1]
        self.temporal_dim = int(temporal_dim)
        if qpos_layout is None or len(qpos_layout) != len(graphs):
            raise ValueError("temporal graph needs one qpos layout per robot graph")
        # per node: column of its joint in the robot's padded qpos vector (-1: no DoF) and the limits
        col, lo, hi = [], [], []
        for g, lay in zip(graphs, qpos_layout):
            pos = {name: k for k, name in enumerate(lay["joints"])}
            for name in g.node_names:
                k = pos.get(name, -1)
                col.append(k)
                lo.append(lay["lower"][k] if k >= 0 else 0.0)
                hi.append(lay["upper"][k] if k >= 0 else 1.0)
        self.register_buffer("q_col", torch.tensor(col), persistent=False)
        self.register_buffer("q_lo", torch.tensor(lo, dtype=torch.float32), persistent=False)
        self.register_buffer("q_hi", torch.tensor(hi, dtype=torch.float32), persistent=False)
        self.node_tenc = TemporalEncoder(2, self.temporal_dim)      # [q, dq] of the node's joint
        self.edge_tenc = TemporalEncoder(4, self.temporal_dim)      # [q, dq] of src and of dst
        d_node += self.temporal_dim + 1                             # + has-DoF flag
        d_edge += self.temporal_dim
        self.in_proj = nn.Linear(d_node, hidden)
        self.layers = nn.ModuleList([MPNNLayer(hidden, d_edge, d_task, layer_norm)
                                     for _ in range(layers)])
        counts = torch.bincount(self.gid, minlength=self.num_graphs).float()
        self.register_buffer("inv_counts", 1.0 / counts.clamp_min(1.0), persistent=False)

    def temporal_features(self, q: torch.Tensor, r_idx: torch.Tensor):
        """q [B, T, Q] raw joint positions of robot r_idx [B] -> node [B, N, dt + 1], edge [B, E, dt] features.
        Nodes of the other robots and nodes without a DoF get zero trajectories and flag 0."""
        qn = q[:, :, self.q_col.clamp_min(0)].float()                                     # [B, T, N]
        qn = (2.0 * (qn - self.q_lo) / (self.q_hi - self.q_lo) - 1.0).clamp(-1.5, 1.5)
        has = ((self.q_col >= 0).unsqueeze(0) & (self.gid.unsqueeze(0) == r_idx.unsqueeze(1))).float()   # [B, N]
        qn = qn * has.unsqueeze(1)
        dq = torch.cat([torch.zeros_like(qn[:, :1]), qn[:, 1:] - qn[:, :-1]], dim=1) * self.DELTA_SCALE
        seq = torch.stack([qn, dq], dim=2).permute(0, 3, 2, 1)                             # [B, N, 2, T]
        xt = torch.cat([self.node_tenc(seq), has.unsqueeze(-1)], dim=-1)
        et = self.edge_tenc(torch.cat([seq[:, self.src], seq[:, self.dst]], dim=2))        # [B, E, dt]
        return xt, et

    def forward(self, e_task: torch.Tensor, q: torch.Tensor = None, r_idx: torch.Tensor = None):
        """Static (zero trajectories): e_task [d_task] (one task) -> g [R, hidden], node embeddings [N, hidden].
        Temporal: e_task [B, d_task], q [B, T, Q], r_idx [B] -> g [B, hidden] of each sample's robot, n [B, N, hidden]."""
        if q is None:
            n = self.in_proj(torch.cat([self.x, self.x.new_zeros(self.x.shape[0], self.temporal_dim + 1)], dim=-1))
            e = torch.cat([self.e, self.e.new_zeros(self.e.shape[0], self.temporal_dim)], dim=-1)
            e_task_nodes = e_task.unsqueeze(0).expand(n.shape[0], -1)
        else:
            xt, et = self.temporal_features(q, r_idx)
            B = q.shape[0]
            n = self.in_proj(torch.cat([self.x.unsqueeze(0).expand(B, -1, -1), xt.to(self.x.dtype)], dim=-1))
            e = torch.cat([self.e.unsqueeze(0).expand(B, -1, -1), et.to(self.e.dtype)], dim=-1)
            e_task_nodes = e_task.unsqueeze(1).expand(-1, n.shape[1], -1)
        for layer in self.layers:
            n = layer(n, self.src, self.dst, e, e_task_nodes)
        if q is None:
            g = torch.zeros(self.num_graphs, n.shape[1], device=n.device, dtype=n.dtype)
            g = g.index_add(0, self.gid, n) * self.inv_counts.unsqueeze(1)
            return g, n
        g = torch.zeros(n.shape[0], self.num_graphs, n.shape[2], device=n.device, dtype=n.dtype)
        g = g.index_add(1, self.gid, n) * self.inv_counts.view(1, -1, 1)
        return g[torch.arange(n.shape[0], device=n.device), r_idx], n


# --------------------------------------------------------------------------- routing
class RoutingCodebook(VectorQuantizer):
    """Codebook addressed by attention: alpha = softmax(s * cos(q, W_K c) / tau), z_soft = alpha @ C, z_hard = C[argmax];
    inference_forward returns (z_hard, idx, -logits) like VectorQuantizer."""

    def __init__(self, num_latents: int, latent_dim: int, temperature: float = 1.0,
                 logit_scale_init: float = 10.0):
        super().__init__(num_latents, latent_dim, code_restart=False)
        nn.init.normal_(self.codebook.weight, std=0.1)
        self.key_proj = nn.Linear(latent_dim, latent_dim, bias=False)
        self.temperature = float(temperature)
        self.log_scale = nn.Parameter(torch.tensor(math.log(logit_scale_init)))

    def logits(self, q: torch.Tensor) -> torch.Tensor:
        keys = self.key_proj(self.codebook.weight)                     # [K, d]
        sim = torch.einsum("...d,kd->...k", F.normalize(q, dim=-1), F.normalize(keys, dim=-1))
        return sim * self.log_scale.exp() / self.temperature

    def forward(self, q: torch.Tensor):
        logits = self.logits(q)
        alpha = logits.softmax(-1)                                     # [..., K]
        z_soft = alpha @ self.codebook.weight
        idx = logits.argmax(-1)
        z_hard = self.codebook(idx)
        if not self.training or self.code_restart:
            self.update_usage(idx)
        return z_soft, z_hard, alpha, idx, logits

    @torch.no_grad()
    def inference_forward(self, q: torch.Tensor):
        logits = self.logits(q)
        idx = logits.argmax(-1)
        return self.codebook(idx), idx, -logits


# --------------------------------------------------------------------------- model
# Options of earlier configs that are now fixed: accepted at this value only (None: any value, unused).
FIXED_OPTIONS = {
    "window_size": None, "max_len": None, "vq_beta": None, "effect_scale": None, "graph_attn_heads": None,
    "depth_enc_low": 0, "depth_dec_low": 0, "use_sinusoidal_pe": False, "add_embodiment_token": False,
    "code_restart": False, "decoder_cond": "graph", "morphology": "graph", "graph_routing": "pool",
    "hard_training": True, "graph_temporal": True,
}


def check_fixed_options(options: dict):
    for k, v in options.items():
        if k not in FIXED_OPTIONS:
            raise TypeError(f"unknown stage-1 option {k!r}")
        if FIXED_OPTIONS[k] is not None and v != FIXED_OPTIONS[k]:
            raise ValueError(f"stage-1 option {k}={v!r} is not supported (only {FIXED_OPTIONS[k]!r})")


class GraphRoutedMotifTokenizer(MotifTokenizer):
    def __init__(
        self,
        state_dim: int = 8,
        model_dim: int = 256,
        latent_dim: int = 256,
        codebook_size: int = 128,
        depth_enc: int = 4,
        depth_dec: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1,
        local_w: int = 16,
        downsample_factor: int = 2,
        # graph / routing
        graph_dir: str = "./assets/kinematic_graphs",
        robots: Dict[str, str] = None,          # graph name -> embodiment tag
        graph_hidden: int = 128,
        graph_layers: int = 3,
        graph_layer_norm: bool = True,
        task_dim: int = 64,
        text_encoder: str = "t5-base",
        route_temperature: float = 1.0,
        graph_temporal_dim: int = 32,
        qpos_layout: str = "./assets/kinematic_graphs/qpos_layout.json",   # data/process/build_qpos_layout.py
        num_primitives: int = 7,
        effect_hidden: int = 256,
        **fixed_options,
    ):
        check_fixed_options(fixed_options)
        super().__init__(
            state_dim=state_dim, model_dim=model_dim, latent_dim=latent_dim,
            codebook_size=codebook_size, depth_enc=depth_enc,
            depth_dec=depth_dec, num_heads=num_heads, dropout=dropout, local_w=local_w,
            downsample_factor=downsample_factor,
        )

        self.forward_for_keys = True
        self.vq = RoutingCodebook(codebook_size, latent_dim, temperature=route_temperature)
        robots = dict(robots or DEFAULT_ROBOTS)
        self.robot_names = list(robots.keys())
        graphs = load_graphs(graph_dir, self.robot_names)
        with open(qpos_layout) as f:
            lay = json.load(f)
        layouts = [lay[r] for r in self.robot_names]
        self.gnn = TaskConditionedMPNN([graphs[r] for r in self.robot_names], task_dim,
                                       hidden=graph_hidden, layers=graph_layers,
                                       layer_norm=graph_layer_norm,
                                       temporal_dim=graph_temporal_dim, qpos_layout=layouts)
        self.task_embed = TaskTextEmbedder(task_dim, text_encoder=text_encoder)
        self.node_names = [f"{r}/{n}" for r in self.robot_names for n in graphs[r].node_names]
        emb2graph = torch.full((max(EMBODIMENT_TAG_MAPPING.values()) + 1,), -1, dtype=torch.long)
        for g_i, (name, tag) in enumerate(robots.items()):
            emb2graph[EMBODIMENT_TAG_MAPPING[tag]] = g_i
        self.register_buffer("emb2graph", emb2graph, persistent=False)
        self.route_query = nn.Linear(graph_hidden + latent_dim, latent_dim)
        self.dec_cond = nn.Linear(graph_hidden, latent_dim)
        self.num_primitives = num_primitives
        # not used by the losses; kept so parameter init and checkpoints match
        self.effect_head = nn.Sequential(nn.Linear(latent_dim, effect_hidden), nn.ReLU(),
                                         nn.Linear(effect_hidden, 3))

    # -- morphology -----------------------------------------------------------
    def morphology_static(self, embodiment_id: torch.Tensor, languages: List[str]):
        """One GNN pass per distinct instruction (zero trajectories), gathered per sample -> g_r [B, Dg]."""
        device = embodiment_id.device
        r_idx = self.emb2graph[embodiment_id.long().view(-1)]
        if (r_idx < 0).any():
            bad = embodiment_id.view(-1)[r_idx < 0].unique().tolist()
            raise ValueError(f"no kinematic graph for embodiment ids {bad}")
        uniq = list(dict.fromkeys(languages))
        e_task = self.task_embed(uniq, device)                          # [U, d_task]
        g_table = torch.stack([self.gnn(e_task[u])[0] for u in range(len(uniq))])   # [U, R, Dg]
        u_idx = torch.tensor([uniq.index(l) for l in languages], device=device)
        return g_table[u_idx, r_idx]

    def morphology_dynamic(self, embodiment_id: torch.Tensor, languages: List[str], qpos: torch.Tensor):
        """g [B, Dg] and node embeddings [B, N, Dg] from the graph carrying the window's qpos [B, T, Q]."""
        device = embodiment_id.device
        r_idx = self.emb2graph[embodiment_id.long().view(-1)]
        e_task = self.task_embed(list(languages), device)               # [B, d_task]
        return self.gnn(e_task, qpos, r_idx)

    # -- forward --------------------------------------------------------------
    def forward(self, batch: Dict[str, torch.Tensor], causal_attn: bool = False):
        progress, x = batch["progress"], batch["rel_state"]
        h = self.encoder(x, progress, causal=causal_attn)               # H_m  [B, M, D]
        g = self.morphology_static(batch["embodiment_id"], batch["language"])  # g_r [B, Dg]
        if "qpos" not in batch:
            raise ValueError("the motion-aware graph needs batch['qpos'] (dataset_*_prim_q)")
        g_route, _ = self.morphology_dynamic(batch["embodiment_id"], batch["language"], batch["qpos"])
        ctx = g_route.unsqueeze(1).expand(-1, h.shape[1], -1)
        q = self.route_query(torch.cat([ctx, h], dim=-1))               # q_m = W_Q[g; h_m]  [B, M, D]
        z_soft, z_hard, alpha, idx, logits = self.vq(q)

        z_dec = z_soft + (z_hard - z_soft).detach()                     # straight-through argmax
        dec_in = z_dec + self.dec_cond(g).unsqueeze(1)
        recon = self.decoder(dec_in, progress, causal=causal_attn)

        ent = -(alpha * (alpha + 1e-10).log()).sum(-1).mean()           # mean token entropy
        mean_alpha = alpha.reshape(-1, alpha.shape[-1]).mean(0)
        balance = (mean_alpha * (mean_alpha * alpha.shape[-1] + 1e-10).log()).sum()  # KL(mean || uniform)
        out = dict(recon=recon, z_e=q, h=h, z_q=z_dec, z=z_hard, idx=idx, alpha=alpha, g=g_route,
                   routing_entropy=ent, load_balance=balance,
                   alpha_max=alpha.max(-1).values.mean().detach())
        prim = batch.get("prim_id")
        if prim is not None:
            out["prim_tok"] = frames_to_tokens(prim.long().view(prim.shape[0], -1),
                                               h.shape[1]).clamp(0, self.num_primitives - 1)
        return out

    def compute_loss(self, x_hat, x) -> Dict[str, torch.Tensor]:
        """Reconstruction only; the routing regularisers are weighted in the trainer
        (lambda_entropy sharpens alpha, lambda_balance spreads codebook usage)."""
        rec = ((x_hat - x) ** 2).mean()
        return {"loss": rec, "recon_loss": rec}


MODEL_REGISTRY = {"Stage1KinematicGraph": GraphRoutedMotifTokenizer}


def build_model(model_cfg: dict):
    name = model_cfg.get("class_name")
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model class_name: {name}")
    kwargs = {k: v for k, v in model_cfg.items() if k != "class_name"}
    return MODEL_REGISTRY[name](**kwargs)
