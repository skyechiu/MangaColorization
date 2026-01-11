"""Model architectures for manga colorization using Pix2Pix GAN.

This module implements the U-Net generator and PatchGAN discriminator
based on the Pix2Pix paper (Isola et al., 2017).
"""

import torch
import torch.nn as nn


class UNetGenerator(nn.Module):
    """U-Net generator for image-to-image translation.

    Takes single-channel grayscale input and produces RGB color output
    using an encoder-decoder architecture with skip connections.
    """

    def __init__(self):
        super().__init__()

        # Encoder path - downsample
        self.enc1 = self._make_encoder_block(1, 64, normalize=False)
        self.enc2 = self._make_encoder_block(64, 128)
        self.enc3 = self._make_encoder_block(128, 256)
        self.enc4 = self._make_encoder_block(256, 512)

        # Bottleneck
        self.bottleneck = self._make_encoder_block(512, 512)

        # Decoder path - upsample with skip connections
        self.dec4 = self._make_decoder_block(512, 512)
        self.dec3 = self._make_decoder_block(1024, 256)
        self.dec2 = self._make_decoder_block(512, 128)
        self.dec1 = self._make_decoder_block(256, 64)

        # Final output layer
        self.final = nn.Sequential(
            nn.ConvTranspose2d(128, 3, kernel_size=4, stride=2, padding=1), nn.Tanh()
        )

    def _make_encoder_block(self, in_channels, out_channels, normalize=True):
        """Create encoder block with conv, optional batchnorm, and activation."""
        layers = [nn.Conv2d(in_channels, out_channels, 4, 2, 1)]
        if normalize:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        return nn.Sequential(*layers)

    def _make_decoder_block(self, in_channels, out_channels):
        """Create decoder block with transposed conv, batchnorm, and activation."""
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        """Forward pass with skip connections."""
        # Encode
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)

        # Bottleneck
        b = self.bottleneck(e4)

        # Decode with skip connections
        d4 = self.dec4(b)
        d4 = torch.cat([d4, e4], dim=1)

        d3 = self.dec3(d4)
        d3 = torch.cat([d3, e3], dim=1)

        d2 = self.dec2(d3)
        d2 = torch.cat([d2, e2], dim=1)

        d1 = self.dec1(d2)
        d1 = torch.cat([d1, e1], dim=1)

        return self.final(d1)


class PatchGANDiscriminator(nn.Module):
    """PatchGAN discriminator for local image classification.

    Classifies 70x70 patches rather than entire images to provide
    more detailed feedback on local structure and textures.
    """

    def __init__(self):
        super().__init__()

        # Input: grayscale (1ch) + color (3ch) = 4 channels
        self.model = nn.Sequential(
            # First layer without batch norm
            nn.Conv2d(4, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            # Subsequent layers with batch norm
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            # Output layer
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1),
        )

    def forward(self, gray, color):
        """Forward pass takes both grayscale and color images."""
        x = torch.cat([gray, color], dim=1)
        return self.model(x)


def test_models():
    """Quick test to verify model shapes work correctly."""
    print("Testing model architectures...")

    # Test generator
    gen = UNetGenerator()
    test_input = torch.randn(1, 1, 256, 256)
    output = gen(test_input)

    assert output.shape == (1, 3, 256, 256), (
        f"Expected (1, 3, 256, 256), got {output.shape}"
    )
    print(f"Generator: {test_input.shape} -> {output.shape}")

    # Test discriminator
    disc = PatchGANDiscriminator()
    gray = torch.randn(1, 1, 256, 256)
    color = torch.randn(1, 3, 256, 256)
    d_output = disc(gray, color)

    print(f"Discriminator: ({gray.shape}, {color.shape}) -> {d_output.shape}")
    print("All tests passed")


if __name__ == "__main__":
    test_models()
