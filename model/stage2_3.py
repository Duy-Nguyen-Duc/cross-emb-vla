import inspect
import json
from pathlib import Path
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from transformers import T5EncoderModel, T5Tokenizer
from transformers.feature_extraction_utils import BatchFeature
from einops import rearrange
from typing import Dict, List

from model.modules.perceiver_resampler_with_mask import PerceiverResampler
from model.modules.flow_matching_action_head import FlowmatchingActionHead

log = logging.getLogger(__name__)

# Use timm's names
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


def _accepted_kwargs(cls, cfg: dict) -> dict:
    # constructor arguments of cls (and of its parents while it forwards **kwargs);
    # old config.json files also carry options that no longer exist
    names = set()
    for klass in cls.__mro__:
        if klass is nn.Module or "__init__" not in vars(klass):
            continue
        params = inspect.signature(klass.__init__).parameters.values()
        names |= {p.name for p in params if p.kind not in (p.VAR_KEYWORD, p.VAR_POSITIONAL)}
        if not any(p.kind == p.VAR_KEYWORD for p in params):
            break
    names.discard("self")
    dropped = sorted(k for k in cfg if k not in names and k != "class_name")
    if dropped:
        log.info("%s: ignoring unused config keys %s", cls.__name__, dropped)
    return {k: v for k, v in cfg.items() if k in names}


class MotifPredictor(nn.Module):
    """Multimodal key learning model using text, vision and Perceiver fusion."""

    def __init__(
        self,
        vq_vae_ckpt: str = None,
        model_dim: int = 512,
        perceiver_depth: int = 6,
        perceiver_heads: int = 8,
        perceiver_dim_head: int = 64,
        perceiver_max_tokens: int = 1000,
    ) -> None:
        super().__init__()

        # Frozen graph-routed stage-1 tokenizer; keys = routing queries q_m from forward(batch).
        from model.stage1_graph import build_model as build_stage1
        vq_vae_cfg_path = Path(vq_vae_ckpt).parent / "config.json"
        with open(vq_vae_cfg_path, "r") as f:
            cfg = json.load(f)
        cfg = cfg['models']['stage1']
        self.vq_vae = build_stage1(cfg)
        checkpoint = torch.load(vq_vae_ckpt, map_location="cpu")
        self.vq_vae.load_state_dict(checkpoint["model"], strict=False)  # the checkpoint also holds the adversarial discriminator
        for param in self.vq_vae.parameters():
            param.requires_grad = False
        self.vq_vae.vq.code_restart = False
        log.info("Loaded VQVAE from %s", vq_vae_ckpt)

        # Perceiver
        code_dim = cfg["latent_dim"]
        ratio = cfg["window_size"] / cfg["downsample_factor"]

        if ratio <= 0 or not float(ratio).is_integer():
            raise ValueError("window_size / downsample_factor must be a positive integer.")

        num_latents = int(ratio)

        self.perceiver = PerceiverResampler(
            dim=model_dim,
            depth=perceiver_depth,
            dim_head=perceiver_dim_head,
            heads=perceiver_heads,
            max_tokens=perceiver_max_tokens,
            num_latents=num_latents,
        )

        self.to_code_dim = nn.Linear(model_dim, code_dim)

        # Vision encoder (DINO)
        dino_dim = 768
        self.dino_transform = transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
        self.dino_encoder = torch.hub.load("facebookresearch/dinov2:main", "dinov2_vitb14_reg")  # pinned ref: loads from the torch hub cache
        self.dino_encoder.requires_grad_(False)
        self.vision_proj = nn.Linear(dino_dim, model_dim)

        # Text encoder (T5)
        t5_dim = 768
        self.text_encoder = T5EncoderModel.from_pretrained("t5-base")
        self.tokenizer = T5Tokenizer.from_pretrained("t5-base")
        self.text_encoder.requires_grad_(False)
        self.lang_proj = nn.Linear(t5_dim, model_dim)

        # (device, sentence) -> unpadded T5 last_hidden_state; plain dict, not in state_dict().
        self._text_cache = {}

        # Frozen encoders stay in eval mode; see train().
        self.dino_encoder.eval()
        self.text_encoder.eval()

    def train(self, mode: bool = True):
        """Keep the frozen encoders in eval mode (no dropout, cacheable text features)."""
        super().train(mode)
        self.dino_encoder.eval()
        self.text_encoder.eval()
        return self

    # Encode text (T5 stays in eval mode, so features are cached per sentence)
    def _encode_text(self, sentences: List[str], device = 'cuda'):
        cache = self._text_cache
        dev_key = str(device)
        missing = [s for s in dict.fromkeys(sentences) if (dev_key, s) not in cache]
        if missing:
            encoding = self.tokenizer(missing, return_tensors="pt", padding=True).to(device)
            with torch.no_grad():
                hidden = self.text_encoder(
                    input_ids=encoding["input_ids"],
                    attention_mask=encoding["attention_mask"],
                ).last_hidden_state
            lengths = encoding["attention_mask"].sum(dim=1).tolist()
            for sentence, row, length in zip(missing, hidden, lengths):
                # Stored unpadded.
                cache[(dev_key, sentence)] = row[:length].clone()

        rows = [cache[(dev_key, s)] for s in sentences]
        max_len = max(row.shape[0] for row in rows)
        lang_emb = rows[0].new_zeros(len(rows), max_len, rows[0].shape[-1])
        attention_mask = torch.zeros(
            len(rows), max_len, dtype=torch.long, device=lang_emb.device
        )
        for i, row in enumerate(rows):
            lang_emb[i, : row.shape[0]] = row
            attention_mask[i, : row.shape[0]] = 1

        # Padded rows stay zero; all consumers mask them out downstream.
        return lang_emb, attention_mask

    # Encode images
    def _encode_image(self, images: torch.Tensor):
        b, v, t, c, h, w = images.shape
        images = rearrange(images, "b v t c h w -> (b t v) c h w")
        images = self.dino_transform(images)

        with torch.no_grad():
            feats = self.dino_encoder.forward_features(images)
            emb = feats["x_norm_patchtokens"]  # [(B*T*V), N, dino_dim]

        emb = rearrange(emb, "(b t v) n d -> b (t v n) d", b=b, t=t, v=v)
        mask = torch.ones(b, emb.shape[1], dtype=torch.int64, device=emb.device)
        return emb, mask

    # Frozen-encoder pass before projection; stage 3 shares it between branches.
    def encode_raw(self, batch: Dict[str, torch.Tensor]):
        lang_emb, lang_mask = self._encode_text(batch["language"], device=batch["images"].device)
        img_emb, img_mask = self._encode_image(batch["images"])
        return lang_emb, lang_mask, img_emb, img_mask

    def _encode_observation_from_raw(self, lang_emb, lang_mask, img_emb, img_mask):
        lang_emb = self.lang_proj(lang_emb)
        img_emb = self.vision_proj(img_emb)

        obs_seq = torch.cat([lang_emb, img_emb], dim=1)
        masks = torch.cat([lang_mask, img_mask], dim=1)
        return obs_seq, masks

    # Forward pass
    def forward(self, batch: Dict[str, torch.Tensor], causal_attn: bool = False):
        obs_seq, masks = self._encode_observation_from_raw(*self.encode_raw(batch))
        key_token = self.perceiver(obs_seq, masks)

        z_e = self.vq_vae(batch, causal_attn=causal_attn)["z_e"]
        key = self.to_code_dim(key_token)
        return F.mse_loss(z_e, key)

    # Inference: key tokens from the frozen-encoder features
    @torch.no_grad()
    def get_key_from_raw(self, lang_emb, lang_mask, img_emb, img_mask):
        obs_seq, masks = self._encode_observation_from_raw(
            lang_emb, lang_mask, img_emb, img_mask
        )
        key_token = self.perceiver(obs_seq, masks)
        return self.to_code_dim(key_token), obs_seq, masks


class MotifConditionedPolicy(nn.Module):
    """Multimodal key learning model using text, vision and Perceiver fusion."""

    def __init__(
        self,
        mkl_cfg: dict = None,
        mkl_ckpt: str = None,
        add_latent_action: bool = False,
        flow_matching_cfg: dict = None,
        geometry_cfg: dict = None,
    ) -> None:
        super().__init__()

        self.add_latent_action = add_latent_action
        if self.add_latent_action and mkl_ckpt is None:
            raise ValueError("Must provide mkl_ckpt if add_latent_action is True.")

        # MotifPredictor (frozen)
        self.mkl = MotifPredictor(**_accepted_kwargs(MotifPredictor, mkl_cfg))

        if mkl_ckpt is not None:
            checkpoint = torch.load(mkl_ckpt, map_location="cpu")
            missing, unexpected = self.mkl.load_state_dict(checkpoint["model"], strict=False)
            for k in missing:
                if not k.startswith(tuple(["vq_vae", "dino_encoder", "text_encoder"])):
                    raise ValueError(f"Missing key: {k}")
                if len(unexpected) > 0:
                    raise ValueError(f"Unexpected keys: {unexpected}")
            log.info(f"Loaded mkl_ckpt from {mkl_ckpt}")

        # Freeze stage2 modules
        for param in self.mkl.parameters():
            param.requires_grad = False

        self.flow_matching_head = FlowmatchingActionHead(flow_matching_cfg)
        self.flow_matching_head.set_trainable_parameters()

        dino_dim = 768
        self.vision_proj = nn.Linear(dino_dim, flow_matching_cfg['hidden_size'])
        t5_dim = 768
        self.lang_proj = nn.Linear(t5_dim, flow_matching_cfg['hidden_size'])

        # Metric 3D positional encodings on the wrist-camera tokens; needs batch["tcp_pose"]. Zero-init.
        self.geometry = None
        if geometry_cfg and geometry_cfg.get("enabled", True):
            from model.geometry_tokens import GeometryTokens
            gcfg = {k: v for k, v in dict(geometry_cfg).items() if k != "enabled"}
            if "robots" in gcfg:
                gcfg["robots"] = dict(gcfg["robots"])
            self.geometry = GeometryTokens(hidden_size=flow_matching_cfg['hidden_size'], **gcfg)

    def get_observation_embedding(self, batch: Dict[str, torch.Tensor], encoded):
        lang_emb, lang_mask, img_emb, img_mask = encoded

        lang_emb = self.lang_proj(lang_emb)
        img_emb = self.vision_proj(img_emb)
        if self.geometry is not None:
            img_emb = self.geometry(img_emb, batch)

        obs_seq = torch.cat([lang_emb, img_emb], dim=1)
        masks = torch.cat([lang_mask, img_mask], dim=1)

        return obs_seq, masks

    def _latent_from_key(self, key: torch.Tensor) -> torch.Tensor:
        """latent_action fed to the flow head: z_q, the quantized motif of the predicted key."""
        z_q, _, _ = self.mkl.vq_vae.vq.inference_forward(key)
        return z_q

    def _head_inputs(self, batch: Dict[str, torch.Tensor]):
        # One shared frozen-encoder pass for the policy branch and mkl's key branch.
        encoded = self.mkl.encode_raw(batch)
        obs_emb, masks = self.get_observation_embedding(batch, encoded=encoded)
        if self.add_latent_action:
            key, _, _ = self.mkl.get_key_from_raw(*encoded)
            latent_action = self._latent_from_key(key)
        else:
            latent_action = None
        backbone = BatchFeature(
            data={
                "backbone_features": obs_emb,
                "backbone_attention_mask": ~masks.bool()    # mask is True for padding tokens
                }
        )
        data = {"embodiment_id": batch['embodiment_id'], "state": batch['state'], "latent_action": latent_action}
        return backbone, data

    def forward(self, batch: Dict[str, torch.Tensor], causal_attn: bool = False):
        backbone, data = self._head_inputs(batch)
        data.update(action=batch['action'], action_mask=batch['action_mask'])
        return self.flow_matching_head(backbone, BatchFeature(data=data)).loss

    @torch.no_grad()
    def get_action(self, batch: Dict[str, torch.Tensor], causal_attn: bool = False):
        backbone, data = self._head_inputs(batch)
        return self.flow_matching_head.get_action(backbone, BatchFeature(data=data))


MODEL_REGISTRY = {
    "Stage2MotifPredictor": MotifPredictor,
    "Stage3Policy": MotifConditionedPolicy,
}

def build_model(model_cfg: dict):
    name = model_cfg.get("class_name")
    if name == "Stage3ELoRAPolicy" and name not in MODEL_REGISTRY:
        from model.stage3_elora import ELoRAPolicy   # lazy: it imports this module
        MODEL_REGISTRY[name] = ELoRAPolicy
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model class_name: {name}")
    return MODEL_REGISTRY[name](**_accepted_kwargs(MODEL_REGISTRY[name], model_cfg))
