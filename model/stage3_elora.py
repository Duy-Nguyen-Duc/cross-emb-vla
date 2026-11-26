import copy
import logging
from typing import Dict, List, Sequence

import contextlib
import warnings

import torch
from torch import nn
from transformers.feature_extraction_utils import BatchFeature

from model.elora import ELoRAContext, ELoRARouting, inject_elora, elora_parameters
from model.stage1_graph import GraphRoutedMotifTokenizer
from model.stage2_3 import MotifConditionedPolicy

log = logging.getLogger(__name__)


@contextlib.contextmanager
def _deterministic_kernels():
    prev, prev_warn = torch.are_deterministic_algorithms_enabled(), torch.is_deterministic_algorithms_warn_only_enabled()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        torch.use_deterministic_algorithms(True, warn_only=True)
        try:
            yield
        finally:
            torch.use_deterministic_algorithms(prev, warn_only=prev_warn)


class GraphELoRARouter(nn.Module):
    """g_r, node embeddings and joint gates from a trainable copy of the stage-1 GNN."""

    def __init__(self, teacher: GraphRoutedMotifTokenizer, gate_hidden: int = 64):
        super().__init__()
        self.gnn = copy.deepcopy(teacher.gnn)
        self.task_embed = copy.deepcopy(teacher.task_embed)
        if getattr(self.task_embed, "proj", None) is not None:
            # Frozen T5 features are identical for both: reuse the teacher's encoder/cache.
            self.task_embed._encode = teacher.task_embed._encode
        hidden, self.num_nodes = self.gnn.in_proj.out_features, int(self.gnn.gid.numel())
        for p in self.parameters():
            p.requires_grad_(True)
        self.register_buffer("emb2graph", teacher.emb2graph.clone(), persistent=False)
        self.gate = nn.Sequential(nn.Linear(hidden, gate_hidden), nn.ReLU(), nn.Linear(gate_hidden, 1))

    def route(self, embodiment_id: torch.Tensor, languages: List[str]):
        """g, gates, node embeddings."""
        device = embodiment_id.device
        r_idx = self.emb2graph[embodiment_id.long().view(-1)]
        if (r_idx < 0).any():
            raise ValueError(f"no kinematic graph for embodiment ids "
                             f"{embodiment_id.view(-1)[r_idx < 0].unique().tolist()}")
        uniq = list(dict.fromkeys(languages))
        e_task = self.task_embed(uniq, device)
        with _deterministic_kernels():
            outs = [self.gnn(e_task[u]) for u in range(len(uniq))]
        g_table = torch.stack([o[0] for o in outs])             # [U, R, Dg]
        n_table = torch.stack([o[1] for o in outs])             # [U, N, Dg]
        u_idx = torch.tensor([uniq.index(l) for l in languages], device=device)
        g = g_table[u_idx, r_idx]                               # [B, Dg]
        n = n_table[u_idx]                                      # [B, N, Dg]
        node_robot = self.gnn.gid
        on_robot = (node_robot.unsqueeze(0) == r_idx.unsqueeze(1)).to(n.dtype)   # [B, N]
        gates = torch.sigmoid(self.gate(n).squeeze(-1)) * on_robot
        return g, gates, n


class ELoRAPolicy(MotifConditionedPolicy):
    def __init__(self, elora_rank: int = 4, elora_targets: Sequence[str] = ("ff.net.0.proj", "ff.net.2"),
                 elora_scale: float = 1.0, elora_gate_hidden: int = 64, **kwargs):
        super().__init__(**kwargs)
        teacher = self.mkl.vq_vae
        if not isinstance(teacher, GraphRoutedMotifTokenizer):
            raise TypeError("Stage3ELoRAPolicy needs a graph-routed stage-1 teacher (Stage1KinematicGraph)")
        self.router = GraphELoRARouter(teacher, gate_hidden=elora_gate_hidden)
        self.elora_ctx = ELoRAContext()
        self.elora_targets = list(elora_targets)
        self.elora_layers = inject_elora(self.flow_matching_head.model, self.elora_targets, self.elora_ctx,
                                         num_skills=teacher.codebook_size, num_nodes=self.router.num_nodes,
                                         rank=elora_rank, scale=elora_scale)
        n_elora = sum(p.numel() for p in elora_parameters(self))
        n_router = sum(p.numel() for p in self.router.parameters())
        log.info("E-LoRA: %d layers, rank %d, %.2fM adapter params, %.2fM router params",
                 len(self.elora_layers), elora_rank, n_elora / 1e6, n_router / 1e6)

    # -- routing ------------------------------------------------------------------
    def _set_routing(self, key: torch.Tensor, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        with torch.no_grad():
            alpha = self.mkl.vq_vae.vq.logits(key).softmax(-1).mean(1)        # [B, K]
        g, gates, _ = self.router.route(batch["embodiment_id"], batch["language"])
        self.elora_ctx.routing = ELoRARouting(alpha=alpha, gate=gates)
        return g

    def _clear_routing(self):
        self.elora_ctx.routing = None

    def _inputs(self, batch, train: bool):
        encoded = self.mkl.encode_raw(batch)
        obs_emb, masks = self.get_observation_embedding(batch, encoded=encoded)
        key, _, _ = self.mkl.get_key_from_raw(*encoded)
        self._set_routing(key, batch)
        latent_action = self._latent_from_key(key)
        inp1 = BatchFeature(data={"backbone_features": obs_emb, "backbone_attention_mask": ~masks.bool()})
        data = {"embodiment_id": batch["embodiment_id"], "state": batch["state"], "latent_action": latent_action}
        if train:
            data.update(action=batch["action"], action_mask=batch["action_mask"])
        return inp1, BatchFeature(data=data)

    def forward(self, batch: Dict[str, torch.Tensor], causal_attn: bool = False):
        if not self.add_latent_action:
            raise ValueError("Stage3ELoRAPolicy needs add_latent_action=true (routing comes from the key)")
        inp1, inp2 = self._inputs(batch, train=True)
        try:
            return self.flow_matching_head(inp1, inp2).loss
        finally:
            self._clear_routing()

    @torch.no_grad()
    def get_action(self, batch: Dict[str, torch.Tensor], causal_attn: bool = False):
        inp1, inp2 = self._inputs(batch, train=False)
        try:
            return self.flow_matching_head.get_action(inp1, inp2)
        finally:
            self._clear_routing()
