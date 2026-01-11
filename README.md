# Pix2Pix-Manga: Automatic Comic Colorization

**Student:** Skye Chiu (s5820249)

## Project Summary

This project implements a Pix2Pix-style conditional GAN for automatic manga
colorization. The system takes a grayscale manga image as input and generates a
plausible RGB colorization.

The goal is not to reproduce the original artist’s colors, but to produce
structurally consistent and visually coherent results that can serve as a
starting point for manual refinement.

---

## Method

- **Generator:** U-Net with skip connections  
- **Discriminator:** 70×70 PatchGAN  
- **Loss:** L1 + adversarial loss (100:1 ratio)

This configuration preserves line art while avoiding desaturated outputs and
remains stable enough to train from scratch within coursework constraints.

---

## Data

- Public anime/manga illustration datasets (mainly Danbooru)
- ~1,000–1,500 color images
- Paired data created by converting color images to grayscale

**Preprocessing**
- Resize to 256×256
- Normalize to [-1, 1]
- Light Gaussian blur to reduce screentone artefacts

**Augmentation**
- Horizontal flip
- Small rotations (±15°)
- Mild color and brightness variation

---

## Training Setup

- Optimizer: Adam (lr=2e-4, betas=0.5, 0.999)
- Batch size: 16
- Epochs: 50
- Hardware: Single GPU (8GB VRAM)

L1 loss weight was increased from 50 to 100 to reduce color bleeding across line
boundaries.

---

## Evaluation

**Metrics**
- SSIM (structural similarity)
- PSNR (pixel-level accuracy)

Metrics were used as guidance only. Visual inspection was necessary, as some
lower-metric outputs appeared perceptually better.

**Best configuration**
- L1 + adversarial loss with skip connections  
- SSIM ~0.75 on validation data

---

## Key Implementation Challenges

**GAN instability (epochs 15–20)**  
Discriminator learned too quickly → generator collapsed.  
Fix: reduced discriminator learning rate (2e-4 → 1e-4).

**Slow data loading**  
On-the-fly grayscale conversion caused major slowdown.  
Fix: preprocessed and cached grayscale images. Epoch time dropped ~30%.

**Color bleeding**  
Colors leaked across edges in early models.  
Fix: increased L1 weight to enforce spatial consistency.

**Memory limits**  
Batch size 32 caused OOM on 8GB VRAM.  
Fix: batch size 16 with no performance loss.

---

## Engineering Practices (Python Coding Standards)

This project follows **PEP 8** and the unit Python coding standards.

### Style and Layout
- 4 spaces indentation (no tabs)
- Max line length: **88**
- Imports grouped as: standard library / third-party / local (blank line between)
- Naming:
  - functions/variables: `lowercase_with_underscores`
  - classes: `CamelCase`
  - constants: `ALL_CAPS`
  - internal helpers: `_leading_underscore`
- Use `is None` / `is not None`, avoid `== None`
- Avoid wildcard imports (`from x import *`)

### Tooling
- **ruff** for linting + formatting (preferred)
- **isort** import sorting (via ruff import rules)
- Project dependencies managed with **uv** (`pyproject.toml` + `uv.lock`)

### Pre-commit
A pre-commit hook is used to ensure formatting and import order before commits.

See `.pre-commit-config.yaml` for:
- `ruff-check` (with import sorting fixes)
- `ruff-format`

`# noqa` is only used when necessary and should be documented inline.

## Ethics

- Public datasets used for educational purposes only
- No real people or personal data involved
- Outputs clearly marked as AI-generated
- Model not released or used commercially

---

## Lessons Learned

- GAN training stability depends more on balance than architecture
- Visual inspection is essential alongside metrics
- Data preprocessing quality matters more than dataset size
- Debugging always takes longer than expected

---

## References

- Isola et al., *Image-to-Image Translation with Conditional Adversarial Networks*
- Ronneberger et al., *U-Net*
- PyTorch Documentation
- Danbooru Dataset
