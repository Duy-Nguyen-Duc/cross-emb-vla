# Decodes episode frames into a cache for data/frame_cache_dataset.py (lossless WebP).
#   python data/process/prextract_frames.py --dataset_root demo_data/maniskill/processed --out_root demo_data/maniskill/frame_cache --format webp

import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from tqdm import tqdm

from utils.video import get_all_frames

VIDEO_KEYS = ["observation.images.image", "observation.images.wrist_image"]
KEY_ALIAS = {"observation.images.image": "image",
             "observation.images.wrist_image": "wrist_image"}
WEBP_LOSSLESS = 101   # cv2: IMWRITE_WEBP_QUALITY > 100 selects lossless encoding


def episode_lengths(dataset_dir: Path) -> dict[int, int]:
    out = {}
    with open(dataset_dir / "meta" / "episodes.jsonl") as f:
        for line in f:
            e = json.loads(line)
            out[int(e["episode_index"])] = int(e["length"])
    return out


def _encode_webp(frame: np.ndarray) -> bytes:
    import cv2
    cv2.setNumThreads(1)
    ok, buf = cv2.imencode(".webp", np.ascontiguousarray(frame), [cv2.IMWRITE_WEBP_QUALITY, WEBP_LOSSLESS])
    assert ok, "webp encode failed"
    return buf.tobytes()


def _episode_frames(dataset_dir: Path, full_key: str, ep: int, n: int) -> np.ndarray:
    frames = get_all_frames(
        (dataset_dir / "videos" / "chunk-000" / full_key / f"episode_{ep:06d}.mp4").as_posix(),
        video_backend="decord", video_backend_kwargs={},
    )
    # clamp: a container can hold a frame more or fewer than the parquet rows
    if frames.shape[0] < n:
        pad = np.repeat(frames[-1:], n - frames.shape[0], axis=0)
        frames = np.concatenate([frames, pad], axis=0)
    return frames[:n]


def extract_robot(dataset_dir: Path, out_dir: Path, workers: int = 8):
    """Cache one robot's frames."""
    out_dir.mkdir(parents=True, exist_ok=True)
    lengths = episode_lengths(dataset_dir)
    eps = sorted(lengths)
    total = sum(lengths[e] for e in eps)

    for full_key in VIDEO_KEYS:
        alias = KEY_ALIAS[full_key]
        meta_path = out_dir / f"frames_{alias}.json"
        if meta_path.exists():
            print(f"  skip {alias} (exists)")
            continue

        # Probe the first episode for the frame shape.
        probe = _episode_frames(dataset_dir, full_key, eps[0], lengths[eps[0]])
        h, w, c = probe.shape[1:]
        shape = (total, h, w, c)
        offsets, cursor = {}, 0

        bin_path = out_dir / f"frames_{alias}.webp.bin"
        idx_path = out_dir / f"frames_{alias}.idx.npy"
        print(f"  {alias}: {total} frames -> lossless webp {shape}, {workers} encoder workers")
        byte_idx = [0]
        with open(bin_path, "wb") as fh, Pool(workers) as pool:
            for ep in tqdm(eps, desc=f"    {alias}", leave=False):
                n = lengths[ep]
                frames = _episode_frames(dataset_dir, full_key, ep, n)
                for blob in pool.imap(_encode_webp, list(frames), chunksize=4):
                    fh.write(blob)
                    byte_idx.append(byte_idx[-1] + len(blob))
                offsets[str(ep)] = [cursor, n]
                cursor += n
        np.save(idx_path.as_posix(), np.asarray(byte_idx, dtype=np.int64))
        meta = {"format": "webp", "shape": list(shape), "offsets": offsets, "video_key": full_key,
                "bytes": byte_idx[-1]}
        print(f"    {alias}: {byte_idx[-1]/1e9:.2f} GB on disk ({byte_idx[-1]/max(total,1)/1024:.1f} KB/frame)")

        assert cursor == total
        with open(meta_path, "w") as f:
            json.dump(meta, f)
        print(f"    wrote {meta_path.name}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_root", required=True,
                   help="parent of the per-robot canonicalized datasets")
    p.add_argument("--out_root", required=True)
    p.add_argument("--robots", nargs="+", default=["panda", "xarm6", "widowxai"])
    p.add_argument("--format", choices=["webp"], default="webp")
    p.add_argument("--workers", type=int, default=8, help="webp encoder processes")
    args = p.parse_args()

    for robot in args.robots:
        print(f"[{robot}]  (all episodes)")
        extract_robot(Path(args.dataset_root) / robot, Path(args.out_root) / robot, workers=args.workers)
    print("PREXTRACT_DONE")


if __name__ == "__main__":
    main()
