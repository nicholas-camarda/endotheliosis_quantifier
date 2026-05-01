"""MPS/CUDA MedSAM HTTP companion for Label Studio box-assist."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from label_studio_converter.brush import mask2rle
from PIL import Image
from segment_anything import sam_model_registry


class CompanionError(RuntimeError):
    """Raised when companion request payload or runtime state is invalid."""


# region agent log
def _agent_debug_log(
    *, run_id: str, hypothesis_id: str, location: str, message: str, data: dict[str, Any]
) -> None:
    try:
        log_path = Path('/Users/ncamarda/Projects/endotheliosis_quantifier/.cursor/debug-b42b0c.log')
        log_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            'sessionId': 'b42b0c',
            'runId': run_id,
            'hypothesisId': hypothesis_id,
            'location': location,
            'message': message,
            'data': data,
            'timestamp': int(time.time() * 1000),
        }
        with log_path.open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(payload, sort_keys=True) + '\n')
    except Exception:
        pass
# endregion


@dataclass
class CompanionModel:
    checkpoint: Path
    device_name: str
    model_type: str = "vit_b"
    threshold: float = 0.5

    def __post_init__(self) -> None:
        self.device = _resolve_device(self.device_name)
        model = sam_model_registry[self.model_type](checkpoint=str(self.checkpoint))
        self.model = model.to(self.device)
        self.model.eval()

    def infer_box(self, *, image_path: Path, box_xyxy: list[float]) -> dict[str, Any]:
        image = Image.open(image_path).convert("RGB")
        width, height = image.size
        image_np = np.array(image)
        # region agent log
        _agent_debug_log(
            run_id='pre-fix',
            hypothesis_id='H6,H7,H8',
            location='src/eq/labelstudio/medsam_companion.py:infer_box.entry',
            message='entered box inference',
            data={
                'image_path': str(image_path),
                'image_size': [int(width), int(height)],
                'box_xyxy': [float(value) for value in box_xyxy],
                'device': self.device_name,
                'model_type': self.model_type,
                'threshold': float(self.threshold),
            },
        )
        # endregion
        image_1024 = np.array(
            image.resize((1024, 1024), resample=Image.Resampling.BICUBIC)
        ).astype(np.float32)
        image_1024 = (image_1024 - image_1024.min()) / np.clip(
            image_1024.max() - image_1024.min(), a_min=1e-8, a_max=None
        )
        tensor = (
            torch.tensor(image_1024)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .to(self.device, dtype=torch.float32)
        )
        with torch.no_grad():
            embedding = self.model.image_encoder(tensor)
            box_np = np.array([box_xyxy], dtype=np.float32)
            box_1024 = box_np / np.array(
                [width, height, width, height], dtype=np.float32
            ) * 1024.0
            box_torch = torch.as_tensor(
                box_1024, dtype=torch.float32, device=self.device
            )[:, None, :]
            sparse, dense = self.model.prompt_encoder(
                points=None, boxes=box_torch, masks=None
            )
            logits, _ = self.model.mask_decoder(
                image_embeddings=embedding,
                image_pe=self.model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse,
                dense_prompt_embeddings=dense,
                multimask_output=False,
            )
            pred = torch.sigmoid(logits)
            pred = F.interpolate(
                pred, size=(height, width), mode="bilinear", align_corners=False
            )
        mask = (pred.squeeze().cpu().numpy() > float(self.threshold)).astype(np.uint8)
        mask_bbox = _binary_mask_bbox(mask)
        box_area = max(0.0, float(box_xyxy[2] - box_xyxy[0])) * max(
            0.0, float(box_xyxy[3] - box_xyxy[1])
        )
        # region agent log
        _agent_debug_log(
            run_id='pre-fix',
            hypothesis_id='H6,H7,H8',
            location='src/eq/labelstudio/medsam_companion.py:infer_box.exit',
            message='finished box inference',
            data={
                'image_path': str(image_path),
                'box_xyxy': [float(value) for value in box_xyxy],
                'box_area': float(box_area),
                'mask_area': int(mask.sum()),
                'mask_bbox': mask_bbox,
                'mask_area_to_box_area': (
                    float(mask.sum()) / float(box_area) if box_area > 0 else None
                ),
            },
        )
        # endregion
        return {
            "width": int(width),
            "height": int(height),
            "rle": mask2rle((mask * 255).astype(np.uint8)),
            "foreground_pixels": int(mask.sum()),
            "model_type": self.model_type,
            "device": self.device_name,
            "image_path": str(image_path),
        }


def _binary_mask_bbox(mask_np: np.ndarray) -> list[int] | None:
    ys, xs = np.where(mask_np > 0)
    if len(xs) == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def _resolve_device(name: str) -> torch.device:
    value = str(name).strip().lower()
    if value == "mps":
        if not torch.backends.mps.is_available():
            raise CompanionError("MPS requested but torch.backends.mps.is_available() is False")
        return torch.device("mps")
    if value == "cuda":
        if not torch.cuda.is_available():
            raise CompanionError("CUDA requested but torch.cuda.is_available() is False")
        return torch.device("cuda")
    if value == "cpu":
        return torch.device("cpu")
    raise CompanionError(f"Unsupported device {name!r}; expected mps|cuda|cpu")

class _Handler(BaseHTTPRequestHandler):
    model: CompanionModel

    def do_GET(self) -> None:  # noqa: N802
        if self.path != "/healthz":
            self._json(HTTPStatus.NOT_FOUND, {"error": "not_found"})
            return
        self._json(
            HTTPStatus.OK,
            {
                "status": "ok",
                "device": self.model.device_name,
                "checkpoint": str(self.model.checkpoint),
                "model_type": self.model.model_type,
            },
        )

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/v1/box_infer":
            self._json(HTTPStatus.NOT_FOUND, {"error": "not_found"})
            return
        try:
            payload = self._read_json()
            image_path = Path(str(payload.get("image_path") or "")).expanduser()
            box = payload.get("box_xyxy")
            if not image_path.exists():
                raise CompanionError(f"image_path does not exist: {image_path}")
            if not isinstance(box, list) or len(box) != 4:
                raise CompanionError("box_xyxy must be [x0, y0, x1, y1]")
            box_xyxy = [float(value) for value in box]
            result = self.model.infer_box(image_path=image_path, box_xyxy=box_xyxy)
        except Exception as exc:  # pragma: no cover - surfaced in integration runs
            self._json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
            return
        self._json(HTTPStatus.OK, result)

    def log_message(self, _format: str, *_args: Any) -> None:  # noqa: A003
        return

    def _read_json(self) -> dict[str, Any]:
        content_length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(content_length) if content_length > 0 else b"{}"
        value = json.loads(body.decode("utf-8"))
        if not isinstance(value, dict):
            raise CompanionError("Expected JSON object payload")
        return value

    def _json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
        encoded = json.dumps(payload).encode("utf-8")
        self.send_response(int(status))
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Serve MedSAM box-assist inference over HTTP."
    )
    parser.add_argument("--checkpoint", required=True, help="Path to MedSAM checkpoint.")
    parser.add_argument(
        "--device",
        choices=["mps", "cuda", "cpu"],
        default="mps",
        help="Inference device. MPS is the default for macOS eq-mac hosts.",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Bind host.")
    parser.add_argument("--port", type=int, default=8098, help="Bind port.")
    parser.add_argument(
        "--model-type",
        default="vit_b",
        choices=["vit_b", "vit_l", "vit_h"],
        help="SAM model registry key.",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.5, help="Mask binarization threshold."
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    model = CompanionModel(
        checkpoint=Path(args.checkpoint).expanduser().resolve(),
        device_name=args.device,
        model_type=args.model_type,
        threshold=args.threshold,
    )
    handler = type("CompanionHandler", (_Handler,), {"model": model})
    server = ThreadingHTTPServer((args.host, int(args.port)), handler)
    print(
        f"MedSAM companion serving on http://{args.host}:{args.port} "
        f"(device={args.device}, checkpoint={model.checkpoint})"
    )
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
