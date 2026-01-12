# Pix2Pix Manga Colorization

**Student:** Skye Chiu (s5820249)  
**Assignment:** ML Programming (NCCA)

## What This Does

Takes grayscale manga pages and colorizes them automatically using a Pix2Pix GAN. The goal isn't to perfectly match the original colors (since we don't know them), but to produce something that looks reasonable and could be used as a starting point for manual coloring.

## Architecture

**Generator:** U-Net with skip connections - preserves line art details  
**Discriminator:** 70×70 PatchGAN - focuses on local texture consistency  
**Loss:** Adversarial + L1 reconstruction (weighted 1:100)

This setup is stable enough to train from scratch and doesn't produce the desaturated outputs you get with pure regression.

## Data

* Source: Danbooru (public anime/manga illustrations)
* Size: ~1000 color images
* Paired data: Created by converting color → grayscale
* Preprocessing: Resize to 256×256, normalize to [-1,1], light Gaussian blur to reduce screentone noise

**Augmentation:**
* Horizontal flip
* Small rotations (±15°)
* Mild color/brightness variation

## Training

* Optimizer: Adam (lr=2e-4, β=(0.5, 0.999))
* Batch size: 16 (32 caused OOM on 8GB VRAM)
* Epochs: 50
* Hardware: Single GTX 1080

I increased L1 weight from 50 to 100 to reduce color bleeding across line boundaries.

## Evaluation

Used SSIM (~0.75 on validation) and PSNR as rough guidelines, but visual inspection was more important. Some lower-metric outputs actually looked better perceptually.

## Problems I Ran Into

**GAN collapse around epoch 15-20:** Discriminator learned too fast and generator got stuck. Fixed by lowering discriminator lr to 1e-4.

**Slow training:** Original code did grayscale conversion on-the-fly which was killing performance. Pre-cached grayscale images and epoch time dropped by about 30%.

**Color bleeding:** Early models had colors leak across edges. Increasing L1 weight helped enforce spatial consistency.

**Memory issues:** Batch size 32 didn't fit in 8GB VRAM. Dropped to 16 with no quality loss.

## Code Standards

This follows NCCA Python coding standards (PEP 8 style):
* 4-space indentation
* Max line length 88 (black/ruff default)
* Imports sorted: stdlib / third-party / local
* Functions/variables: `snake_case`
* Classes: `CamelCase`

Dependencies managed with `uv` (see `pyproject.toml`). Pre-commit hooks run ruff for formatting and linting.

## Ethics

* Used public datasets for educational purposes only
* No personal data or identifiable people
* Outputs marked as AI-generated
* Not intended for commercial use

## What I Learned

Getting the discriminator/generator balance right was way more important than I expected - small LR changes had huge effects. Also learned that metrics don't always correlate with visual quality for colorization, and that data preprocessing quality matters more than just having a huge dataset. Oh and debugging GAN training takes forever.

## References

* Isola et al. (2017). Image-to-Image Translation with Conditional Adversarial Networks. CVPR.
* Ronneberger et al. (2015). U-Net: Convolutional Networks for Biomedical Image Segmentation. MICCAI.
* Danbooru Dataset: https://danbooru.donmai.us/
