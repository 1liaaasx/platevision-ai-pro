# Model cards and use conditions

## Shared system context

PlateVision AI Pro is an experimental reader for single-line alphanumeric plates. It is intended for education and qualitative demonstration. It should not be used for automated enforcement, access decisions, identity verification or surveillance without a separately validated system and applicable permissions. Images may contain personal data; the app processes them on the server where it runs and the release excludes the private demonstration photo. Users control retention of their own uploads and exports.

| Model | Input / output | Dataset and evaluation | Main limits |
| --- | --- | --- | --- |
| CNN V1, historical baseline | Isolated 32×32 grayscale glyph → one of 36 classes; original checkpoint unavailable | Supplied synthetic test: 57.36% character accuracy, 5,188 test items shared with the V2 historical experiment | Architecture, split and provenance are incomplete; no V1 inference in the app |
| CNN V2, `platevision_cnn_v2.keras` | Bright glyph on dark 32×32 float32 canvas → 36 raw softmax values | Supplied synthetic historical test: 80.65% character accuracy, loss 1.0574; no independent real-photo benchmark | Weak real-photo glyph generalization; detection and segmentation errors; historical font split unknown |
| CNN V3.1, `platevision_cnn_v31.keras` | Bright glyph on dark 32×32 float32 canvas after its own 48×48 reconstruction → 36 raw softmax values | Synthetic Lato validation: 98.17% accuracy, 98.15% macro F1; reserved DejaVu test not evaluated | Domain shift from synthetic render; `0/O` confusion (zero recall 67% in Lato validation); no measured real-photo population accuracy |
| EasyOCR 1.7.2, optional | Prepared crop (gray, ×4, Otsu) → text and raw OCR scores | One private demonstration photo transcribed exactly; no representative benchmark | Large CPU and memory cost; may fail on unfamiliar plates; score uncalibrated; external official weights required |

## Risk and interpretation

A high character-level result does not imply accurate full-plate reading. A proposed string may be wrong even when raw softmax values are high. Reference comparison is post hoc and never adjusts predictions. A private development photo was used for qualitative checks; its image, plate text and detailed outputs are excluded from this candidate. That observation does not establish an OCR success rate. The automatic OpenCV detector can fail on small, oblique, blurred, reflective and partially visible regions. Manual crop is a fallback, not a validation of the transcription.

## Reuse and redistribution

The two custom checkpoints were supplied for this local project. Their public redistribution terms have not been independently established; confirm ownership and permission before publishing model files. EasyOCR is a separately installed package with external weights; follow its upstream terms and the local checksum-verified download procedure. Synthetic screenshots in this release contain no private vehicle photo. No project-wide LICENSE is included pending the owner's choice and rights review.
