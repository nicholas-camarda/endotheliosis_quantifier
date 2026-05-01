# Label Studio MedSAM Companion (MPS host)

This companion exposes real MedSAM box-prompt inference over HTTP for hybrid Label Studio grading.

## Runtime contract

- `GET /healthz`
  - Returns `200` with `{status, device, checkpoint, model_type}` when model is loaded.
- `POST /v1/box_infer`
  - Request JSON:
    - `image_path`: absolute path to source image on host filesystem
    - `box_xyxy`: `[x0, y0, x1, y1]` in original pixel coordinates
  - Response JSON:
    - `width`, `height`
    - `rle` (binary run-length payload for `brushlabels`)
    - `foreground_pixels`
    - `model_type`, `device`, `image_path`

## Launch on macOS (`eq-mac`, MPS)

```bash
conda activate eq-mac
env PYTORCH_ENABLE_MPS_FALLBACK=1 \
  /Users/ncamarda/mambaforge/envs/eq-mac/bin/python \
  -m eq.labelstudio.medsam_companion \
  --checkpoint /absolute/path/to/medsam_glomeruli_best_sam_state_dict.pth \
  --device mps \
  --host 127.0.0.1 \
  --port 8098
```

Configure `configs/label_studio_medsam_hybrid.yaml` to point to the same base URL and health path.

## Notes

- Keep the companion on the host when using MPS; Docker containers on macOS do not expose Metal/MPS acceleration for this workflow.
- The Label Studio Docker container continues to run separately via `eq labelstudio start`.
- Do not commit checkpoints or runtime logs to Git.
