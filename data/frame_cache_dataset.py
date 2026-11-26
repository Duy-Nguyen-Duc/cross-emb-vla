

import json
from pathlib import Path

import numpy as np

from .dataset import LeRobotSingleDataset

# reads frames from a precomputed webp cache instead of mp4s; step index == frame
class FrameCachedLeRobotSingleDataset(LeRobotSingleDataset):
    def __init__(self, *args, frame_cache_dir: str | Path, **kwargs):
        self._frame_cache_dir = Path(frame_cache_dir)
        super().__init__(*args, **kwargs)
        self._blob = {}       # alias -> (byte memmap, int64 index, HWC)
        self._offsets = {}
        for key in self.modality_keys.get("video", []):
            alias = key.replace("video.", "")
            meta_path = self._frame_cache_dir / f"frames_{alias}.json"
            if not meta_path.exists():
                raise FileNotFoundError(
                    f"no frame cache for {alias} at {meta_path}; run "
                    f"data/process/prextract_frames.py first"
                )
            meta = json.load(open(meta_path))
            self._offsets[alias] = {int(k): tuple(v) for k, v in meta["offsets"].items()}
            if meta.get("format") != "webp":
                raise ValueError(f"frame cache {meta_path}: expected format webp")
            blob = np.memmap((self._frame_cache_dir / f"frames_{alias}.webp.bin").as_posix(),
                             dtype=np.uint8, mode="r")
            idx = np.load((self._frame_cache_dir / f"frames_{alias}.idx.npy").as_posix())
            self._blob[alias] = (blob, idx, tuple(meta["shape"][1:]))

    def _decode_webp(self, alias: str, idx: np.ndarray) -> np.ndarray:
        import cv2
        blob, byte_idx, hwc = self._blob[alias]
        out = np.empty((len(idx),) + hwc, dtype=np.uint8)
        for j, i in enumerate(idx):
            buf = np.asarray(blob[byte_idx[i]:byte_idx[i + 1]])
            img = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
            if img is None or img.shape != hwc:
                raise RuntimeError(f"frame cache {alias}: bad webp frame {i}")
            out[j] = img
        return out

    def get_video(self, trajectory_id: int, key: str, base_index: int) -> np.ndarray:
        step_indices = self.delta_indices[key] + base_index
        trajectory_index = self.get_trajectory_index(trajectory_id)
        # same clamping as the parent
        step_indices = np.maximum(step_indices, 0)
        step_indices = np.minimum(step_indices, self.trajectory_lengths[trajectory_index] - 1)
        assert key.startswith("video."), f"Video key must start with 'video.', got {key}"
        alias = key.replace("video.", "")
        start, length = self._offsets[alias][int(trajectory_id)]
        idx = start + np.minimum(step_indices, length - 1)
        return self._decode_webp(alias, idx)
