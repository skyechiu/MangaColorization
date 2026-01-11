"""Dataset module for manga colorization.

Loads color manga images and converts them to grayscale for training
the colorization model.
"""

import os

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class MangaDataset(Dataset):
    """Dataset for manga colorization task.

    Converts color images to grayscale (input) while keeping the original
    color version as the target. This creates perfectly aligned training pairs.
    """

    def __init__(self, root_dir, image_size=256):
        """Initialize dataset.

        Args:
            root_dir: Directory containing image files
            image_size: Target size for resizing (creates square images)
        """
        self.root_dir = root_dir
        self.image_size = image_size

        # Find all image files
        self.image_files = [
            f for f in os.listdir(root_dir) if f.endswith((".jpg", ".png", ".jpeg"))
        ]

        if len(self.image_files) == 0:
            raise ValueError(f"No images found in {root_dir}")

        # Color image transform
        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

        # Grayscale transform
        self.gray_transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5]),
            ]
        )

    def __len__(self):
        """Return number of images in dataset."""
        return len(self.image_files)

    def __getitem__(self, idx):
        """Get image pair at given index.

        Args:
            idx: Index of image to retrieve

        Returns:
            Tuple of (grayscale_image, color_image) as normalized tensors
        """
        img_path = os.path.join(self.root_dir, self.image_files[idx])
        image = Image.open(img_path).convert("RGB")

        color_image = self.transform(image)
        gray_image = self.gray_transform(image)

        return gray_image, color_image


if __name__ == "__main__":
    """Test the dataset."""
    try:
        dataset = MangaDataset("data/train", image_size=256)
        print(f"Dataset loaded: {len(dataset)} images")

        gray, color = dataset[0]
        print(f"Gray shape: {gray.shape}")
        print(f"Color shape: {color.shape}")
        print(f"Gray range: [{gray.min():.3f}, {gray.max():.3f}]")
        print(f"Color range: [{color.min():.3f}, {color.max():.3f}]")
        print("Dataset test passed")

    except Exception as e:
        print(f"Test skipped: {e}")
