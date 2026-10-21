# Real photo benchmark

Place only photos you are authorized to process in `images/`. Keep them private unless redistribution rights are explicit. Copy `annotations.example.json` to `annotations.json` and add entries with a unique `id`, filename under `images/`, `plate_bbox` as `[left, top, right, bottom]` in original pixel coordinates, uppercase `text` without spaces, and a nonempty `rights` note. The reference is used only for evaluation.

Run from the project root:

```powershell
python -m scripts.benchmark_real --annotations benchmark_real/annotations.json --images-dir benchmark_real/images --output benchmark_real/report.json --engines "CNN V2" "CNN V3.1" "EasyOCR"
```

The evaluator measures automatic detection IoU and recall at 0.5, exact whole-plate accuracy, edit-distance character accuracy, and processing time. It records each failure. The example annotation set is intentionally empty; no real-photo aggregate score is available until an authorized, independently annotated dataset is supplied. The private `images.jpg` demonstration is not shipped here. Results and images under this directory are excluded from distribution by default.
