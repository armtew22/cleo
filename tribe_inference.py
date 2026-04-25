"""Pluggable TRIBE v2 inference module.

Ports:
  - inputs: a 30s video file (audio + auto-transcribed text are extracted
            internally), or explicit (video_path, audio_path, text) components.
  - output: a numpy array of shape (T, V) where T is the number of segments
            and V=20484 is the fsaverage5 cortical-vertex count, plus the
            list of Segment objects describing per-row stimulus timing.

Typical use from a control unit:
    from tribe_inference import TribeInference
    engine = TribeInference()              # one-time model load
    preds, segments = engine.predict("clip_30s.mp4")
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch


DEFAULT_HF_HOME = "/home/md2292/tribev2-bench/hfcache"
DEFAULT_CACHE = "/home/md2292/tribev2-bench/cache"
DEFAULT_HF_TOKEN_FILE = "/share/goyal/lio/huggingface/token"


@dataclass
class TribeOutput:
    preds: np.ndarray            # (T, 20484) float32
    segments: list               # list[neuralset.events.Segment]

    @property
    def shape(self) -> tuple[int, int]:
        return tuple(self.preds.shape)


class TribeInference:
    """Loads TRIBE v2 once and exposes a simple predict() port."""

    def __init__(
        self,
        device: str = "cuda",
        hf_home: str = DEFAULT_HF_HOME,
        cache_folder: str = DEFAULT_CACHE,
        hf_token_file: Optional[str] = DEFAULT_HF_TOKEN_FILE,
        model_id: str = "facebook/tribev2",
    ):
        os.environ.setdefault("HF_HOME", hf_home)
        os.environ.setdefault("HF_HUB_CACHE", f"{hf_home}/hub")
        os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
        if hf_token_file and Path(hf_token_file).exists() and "HF_TOKEN" not in os.environ:
            tok = Path(hf_token_file).read_text().strip()
            os.environ["HF_TOKEN"] = tok
            os.environ["HUGGINGFACE_HUB_TOKEN"] = tok

        from tribev2 import TribeModel  # imported after env is set

        Path(cache_folder).mkdir(parents=True, exist_ok=True)
        self.device = device
        self._model = TribeModel.from_pretrained(model_id, cache_folder=cache_folder)
        self._model._model.eval()

    def predict(self, video_path: str | os.PathLike) -> TribeOutput:
        """Run inference on a 30s video file (audio + text auto-extracted)."""
        video_path = str(video_path)
        events = self._model.get_events_dataframe(video_path=video_path)
        with torch.inference_mode():
            preds, segments = self._model.predict(events=events, verbose=False)
        arr = preds.detach().cpu().numpy() if hasattr(preds, "detach") else np.asarray(preds)
        return TribeOutput(preds=arr.astype(np.float32, copy=False), segments=list(segments))

    def predict_timed(self, video_path: str | os.PathLike) -> tuple[TribeOutput, float, float]:
        """Same as predict() but also returns (latency_seconds, peak_vram_gb).

        Discipline: torch.cuda.synchronize before/after, peak-mem stats reset.
        """
        import time
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = self.predict(video_path)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        peak_gb = torch.cuda.max_memory_allocated() / 1e9
        return out, dt, peak_gb


if __name__ == "__main__":
    import argparse, json
    p = argparse.ArgumentParser(description="Run TRIBE v2 on a 30s video.")
    p.add_argument("video", help="Path to a video file with audio (>=30s ideal).")
    p.add_argument("--save-npy", help="Write preds array to this .npy path.")
    args = p.parse_args()

    engine = TribeInference()
    out, dt, peak = engine.predict_timed(args.video)
    print(json.dumps({
        "video": args.video,
        "out_shape": list(out.shape),
        "n_segments": len(out.segments),
        "latency_s": round(dt, 4),
        "peak_vram_gb": round(peak, 3),
    }, indent=2))
    if args.save_npy:
        np.save(args.save_npy, out.preds)
        print(f"wrote {args.save_npy}")
