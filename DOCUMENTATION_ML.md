# Machine-learning methodology

## Task and classes

The CNN task is **isolated character classification**, not full-plate recognition. The class order is exactly `0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ` (36 labels). Both checkpoints accept `float32` tensors shaped `(batch, 32, 32, 1)` with bright foreground on a dark background, and output 36 softmax values. The plate-level pipeline adds OpenCV detection and segmentation; errors in either step propagate to every recognition engine.

## Datasets and generation

The historical V1/V2 dataset is described only as synthetic alphanumeric glyphs. Its supplied test has 5,188 examples, but the original font list, split construction, seeds and augmentation settings are not fully documented. The original V2 training notebook is not part of this publication candidate; the supplied historical scores have not been independently reproduced here.

V3.1 generated 21,600 synthetic training, 3,600 Lato-family validation and 3,600 **reserved** DejaVu-family test characters, according to `data/manifest_v31.json`. Training groups were Arimo, Carlito, Cousine, FreeSans, Liberation and Roboto. The manifest records individual font SHA-256 hashes, seed `2026`, and validation data seed `2228`. The generator draws dark text on a light 48×48 canvas, varies font size, position and ink, then applies the documented moderate morphology/shear and resizes to 32×32. Training used uniform group sampling and increased class weights for `1`, `I`, `U` and `J`. The full original training environment is not bundled. A family-separated split reduces direct font leakage but does not guarantee visual independence among related designs.

## Models and inference contract

- **V1:** historical baseline with a supplied result but no independently verified original checkpoint or architecture.
- **V2:** supplied Keras CNN with four convolution layers (32, 64, 128, 128 filters), batch normalization, two max-pooling operations, 128-unit dense layer and 0.5 dropout before 36-class softmax. Segmented binary glyphs are centered in a 32×32 dark canvas and scaled into `[0,1]` by `character_tensor`.
- **V3.1:** supplied 508,228-parameter Keras CNN, independent from V2. For photo inference, `v31_character_tensor` reconstructs a light 48×48 canvas around the segmented glyph, reduces to 32×32 and inverts to bright foreground. The training render included actual antialiasing, font shape and augmentation; the reconstructed photo mask cannot exactly reproduce it. The checkpoint is frozen and its SHA-256 is verified before loading.
- **EasyOCR:** independent CRAFT detector and English G2 recognizer, optional on CPU. The raw RGB crop is converted to gray, enlarged ×4, then Otsu binarized. Its model files are checked against the EasyOCR 1.7.2 official checksums before reader construction. The app does not download them during inference.

## Supplied results and interpretation

| Protocol | V1 | V2 | V3.1 |
| --- | ---: | ---: | ---: |
| Historical synthetic character test | 57.36% accuracy | 80.65% accuracy; loss 1.0574 | — |
| Synthetic Lato character validation | — | — | 98.17% accuracy; 98.15% macro F1; loss 0.0561 |
| Reserved DejaVu character test | — | — | Not evaluated |

The V2 test and V3.1 validation use different datasets and cannot establish a 17.52-point gain. In V3.1's Lato validation, `0` recall was **67%**, with **33 of 100** zeros classified as `O`. `J` recall was 100% in that validation. A private real-photo example was used for development checks. Its contents and outputs are excluded, and it cannot estimate any population accuracy.

CNN softmax maxima and EasyOCR scores are raw outputs, not calibrated probabilities of whole-plate correctness. A reference entered in the UI is applied only after predictions. No rule silently replaces `0/O`, `8/B`, `J/U` or `1/I`.

## Domain shift and methodological limits

Synthetic character accuracy excludes plate detection, perspective, low resolution, blur, glare, partial occlusion, image compression, nonmatching fonts and segmentation mistakes. Enlarging a small plate crop aids processing but cannot recover missing detail. A valid real-photo benchmark needs independent authorized photos, fixed annotations, an untouched evaluation set and separate detection, exact-plate and character metrics. `benchmark_real/` provides the infrastructure; no such aggregate score has been claimed.
