import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# WHY THESE SPECIFIC SETTINGS FOR YOUR SETUP:
#
# REFUGE Dataset: 1200 images (400 normal, 800 glaucoma — IMBALANCED!)
# CPU Training:   Small batch size (8) to avoid RAM overload
# Image size:     224x224 (ResNet-50 requirement)
# ─────────────────────────────────────────────────────────────────────────────

# ── CLASS LABELS ─────────────────────────────────────────────────────────────
# normal   → label 0
# glaucoma → label 1
CLASS_NAMES = ['normal', 'glaucoma']

# ── DATA AUGMENTATION ────────────────────────────────────────────────────────
# TRAINING transforms: augmentation + normalization
# WHY AUGMENTATION?
# - We only have ~840 training images (70% of 1200)
# - Augmentation artificially increases variety
# - Prevents overfitting (model memorizing exact images)
# - Each epoch the model sees SLIGHTLY different versions of images

train_transforms = transforms.Compose([
    # Resize to ResNet-50 input size
    transforms.Resize((224, 224)),

    # Random horizontal flip (50% chance)
    # WHY: Fundus images can be mirrored — both are valid
    transforms.RandomHorizontalFlip(p=0.5),

    # Random vertical flip (50% chance)
    # WHY: Valid augmentation for fundus images
    transforms.RandomVerticalFlip(p=0.5),

    # Random rotation up to 30 degrees
    # WHY: Camera angle varies between patients
    transforms.RandomRotation(degrees=30),

    # Random brightness and contrast changes
    # WHY: Different cameras/lighting conditions
    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.1,
        hue=0.05
    ),

    # Convert PIL image to tensor (values become 0-1)
    transforms.ToTensor(),

    # Normalize using ImageNet mean and std
    # WHY: ResNet-50 was pretrained on ImageNet with these exact values
    # Using same normalization = better transfer learning
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  # ImageNet mean (R, G, B)
        std=[0.229, 0.224, 0.225]    # ImageNet std  (R, G, B)
    )
])

# VALIDATION and TEST transforms: NO augmentation, only resize + normalize
# WHY NO AUGMENTATION HERE?
# - Val/Test must reflect real-world conditions
# - Augmenting test data would give unreliable accuracy numbers
val_test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ── CUSTOM DATASET CLASS ──────────────────────────────────────────────────────
class GlaucomaDataset(Dataset):
    """
    Custom PyTorch Dataset for loading glaucoma fundus images.

    WHAT IS A DATASET CLASS?
    - PyTorch needs a special class to load your images.
    - It tells PyTorch: "here's how to load one image and its label"
    - PyTorch then handles batching, shuffling automatically.

    Folder structure expected:
        root_dir/
            normal/    ← images labeled 0
            glaucoma/  ← images labeled 1
    """

    def __init__(self, root_dir, transform=None):
        """
        root_dir: path to train/, val/, or test/ folder
        transform: augmentation/normalization pipeline
        """
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []   # list of (image_path, label) tuples
        self.labels = []    # list of labels only (needed for class weights)

        # Scan folders and collect all image paths + labels
        for label_idx, class_name in enumerate(CLASS_NAMES):
            class_folder = os.path.join(root_dir, class_name)

            if not os.path.exists(class_folder):
                print(f"⚠️  Warning: {class_folder} not found!")
                continue

            valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
            images_found = 0

            for filename in sorted(os.listdir(class_folder)):
                if filename.lower().endswith(valid_extensions):
                    img_path = os.path.join(class_folder, filename)
                    self.samples.append((img_path, label_idx))
                    self.labels.append(label_idx)
                    images_found += 1

            print(f"  Found {images_found} images in '{class_name}/' "
                  f"(label={label_idx})")

        print(f"  Total: {len(self.samples)} images loaded from {root_dir}\n")

    def __len__(self):
        """Returns total number of images — PyTorch requires this"""
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Load and return ONE image + its label.
        PyTorch calls this automatically during training.

        idx: index of the image (0 to len-1)
        Returns: (image_tensor, label)
        """
        img_path, label = self.samples[idx]

        # Load image as RGB (3 channels — required by ResNet-50)
        image = Image.open(img_path).convert('RGB')

        # Apply transforms (augmentation + normalization)
        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.float32)

    def get_class_weights(self):
        """
        Compute class weights to handle REFUGE dataset imbalance.

        REFUGE has: 400 normal, 800 glaucoma → IMBALANCED!
        Without class weights, model will be biased toward glaucoma.

        Formula: weight[class] = total_samples / (n_classes * class_count)
        Higher weight = model pays MORE attention to that class.
        """
        labels = np.array(self.labels)
        total = len(labels)
        n_classes = len(CLASS_NAMES)

        weights = []
        for i in range(n_classes):
            count = np.sum(labels == i)
            weight = total / (n_classes * count)
            weights.append(weight)
            print(f"  Class '{CLASS_NAMES[i]}': {count} samples, "
                  f"weight = {weight:.4f}")

        return torch.tensor(weights, dtype=torch.float32)


# ── DATALOADER CREATION ───────────────────────────────────────────────────────
def create_dataloaders(dataset_root, batch_size=8, num_workers=0):
    """
    Create train, validation, and test DataLoaders.

    WHAT IS A DATALOADER?
    - Wraps your Dataset and feeds images to the model in BATCHES
    - Handles shuffling, parallel loading, etc.

    WHY BATCH SIZE = 8 FOR CPU?
    - CPU has limited RAM (typically 8-16GB)
    - Larger batch = more RAM used
    - batch_size=8 is safe for CPU with 224x224 images
    - On GPU you could use 32-64

    WHY num_workers=0 FOR CPU?
    - num_workers controls parallel data loading
    - On Windows CPU, num_workers > 0 can cause errors
    - Set to 0 to be safe (load data in main thread)

    WHY shuffle=True FOR TRAIN ONLY?
    - Training: shuffle so model doesn't learn order of images
    - Val/Test: no shuffle (consistent evaluation)
    """

    print("="*50)
    print("LOADING DATASETS")
    print("="*50)

    # Create dataset objects
    print("\n📁 Training set:")
    train_dataset = GlaucomaDataset(
        os.path.join(dataset_root, 'train'),
        transform=train_transforms
    )

    print("📁 Validation set:")
    val_dataset = GlaucomaDataset(
        os.path.join(dataset_root, 'val'),
        transform=val_test_transforms
    )

    print("📁 Test set:")
    test_dataset = GlaucomaDataset(
        os.path.join(dataset_root, 'test'),
        transform=val_test_transforms
    )

    # Compute class weights from training set
    print("\n⚖️  Computing class weights (for imbalanced REFUGE dataset):")
    class_weights = train_dataset.get_class_weights()

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,        # Shuffle training data every epoch
        num_workers=num_workers,
        pin_memory=False     # Only True for GPU
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False
    )

    # Print summary
    print("\n" + "="*50)
    print("DATALOADER SUMMARY")
    print("="*50)
    print(f"  Batch size:          {batch_size}")
    print(f"  Training batches:    {len(train_loader)}")
    print(f"  Validation batches:  {len(val_loader)}")
    print(f"  Test batches:        {len(test_loader)}")
    print(f"  Total train images:  {len(train_dataset)}")
    print(f"  Total val images:    {len(val_dataset)}")
    print(f"  Total test images:   {len(test_dataset)}")

    return train_loader, val_loader, test_loader, class_weights


def visualize_batch(dataloader, num_images=8):
    """
    Show a sample batch of images to verify loading is correct.
    Run this BEFORE training to make sure images look right.
    """
    images, labels = next(iter(dataloader))

    # Denormalize for display
    # (undo the ImageNet normalization we applied)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    for i in range(min(num_images, len(images))):
        img = images[i] * std + mean       # Denormalize
        img = img.permute(1, 2, 0).numpy() # CHW → HWC for matplotlib
        img = np.clip(img, 0, 1)

        label = int(labels[i].item())
        class_name = CLASS_NAMES[label]
        color = 'red' if label == 1 else 'green'

        axes[i].imshow(img)
        axes[i].set_title(f'{class_name.upper()}',
                         color=color, fontweight='bold', fontsize=11)
        axes[i].axis('off')

    plt.suptitle('Sample Training Batch\n'
                 '(Green = Normal, Red = Glaucoma)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig('batch_sample.png', dpi=100, bbox_inches='tight')
    print(f"Image tensor shape: {images.shape}")
    print(f"Label tensor shape: {labels.shape}")
    print("✅ Batch visualization saved to batch_sample.png")


# ── TEST IT ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    train_loader, val_loader, test_loader, class_weights = create_dataloaders(
        dataset_root="dataset",
        batch_size=8
    )

    print("\n🖼️  Visualizing a sample batch...")
    visualize_batch(train_loader)
    print("\n✅ Dataset preparation complete!")
