"""
MRL Eye Dataset Augmentation
Each image generates exactly 5 augmented versions:
  1. Horizontal Flip
  2. Rotation +20 degrees
  3. Rotation -20 degrees
  4. Brightness + Contrast boost
  5. Combination (flip + rotation + noise)

Usage:
    python augment_mrl.py
"""

import os
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from pathlib import Path
import shutil
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# =============================================================================
# CONFIGURATION
# =============================================================================

class Config:
    
    # Input dataset
    INPUT_DIR  = "data/MRL"
    
    # Output directory
    OUTPUT_DIR = "data/MRL_AUGMENTED"
    
    # Only augment training set
    AUGMENT_SPLITS = ["train"]
    
    # Copy val and test as-is
    COPY_SPLITS = ["val", "test"]
    
    # Random seed
    SEED = 42
    
    # Image size
    IMAGE_SIZE = (224, 224)


# =============================================================================
# 5 AUGMENTATION FUNCTIONS
# =============================================================================

def augment_1_horizontal_flip(img):
    """
    PART 1: Horizontal Flip
    Simulates left/right eye symmetry
    
    Before: →_→  (left eye looking right)
    After:  ←_←  (mirrored)
    """
    return img.transpose(Image.FLIP_LEFT_RIGHT)


def augment_2_rotate_positive(img):
    """
    PART 2: Rotate +20 degrees (clockwise)
    Simulates head tilt to the right
    
    Before: ___
    After:   \  (tilted right)
    """
    return img.rotate(
        -20,                        # Negative = clockwise
        resample=Image.BILINEAR,
        expand=False,
        fillcolor=(0, 0, 0)         # Black fill for corners
    )


def augment_3_rotate_negative(img):
    """
    PART 3: Rotate -20 degrees (counter-clockwise)
    Simulates head tilt to the left
    
    Before: ___
    After:  /   (tilted left)
    """
    return img.rotate(
        20,                         # Positive = counter-clockwise
        resample=Image.BILINEAR,
        expand=False,
        fillcolor=(0, 0, 0)
    )


def augment_4_brightness_contrast(img):
    """
    PART 4: Brightness + Contrast adjustment
    Simulates different lighting environments:
      - Daytime driving
      - Night driving
      - Tunnel lighting
    """
    # Randomly choose bright or dark variant
    import random
    random.seed(None)  # True random each call
    
    choice = random.choice(['bright', 'dark', 'high_contrast'])
    
    if choice == 'bright':
        # Simulate bright daylight
        img = ImageEnhance.Brightness(img).enhance(1.4)
        img = ImageEnhance.Contrast(img).enhance(1.2)
    
    elif choice == 'dark':
        # Simulate night driving
        img = ImageEnhance.Brightness(img).enhance(0.6)
        img = ImageEnhance.Contrast(img).enhance(1.3)
    
    else:  # high_contrast
        # Simulate harsh lighting (tunnel exit)
        img = ImageEnhance.Contrast(img).enhance(1.5)
        img = ImageEnhance.Brightness(img).enhance(1.1)
    
    return img


def augment_5_combination(img):
    """
    PART 5: Combination augmentation
    Applies multiple small transforms together:
      - Small flip (50% chance)
      - Small rotation (±10 degrees)
      - Slight blur (camera shake)
      - Slight noise (sensor noise)
      - Small crop + resize
    """
    import random
    random.seed(None)
    
    # Small horizontal flip (50% chance)
    if random.random() > 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    
    # Small rotation (±10 degrees)
    angle = random.uniform(-10, 10)
    img = img.rotate(angle, resample=Image.BILINEAR, fillcolor=(0, 0, 0))
    
    # Slight color jitter
    img = ImageEnhance.Brightness(img).enhance(random.uniform(0.85, 1.15))
    img = ImageEnhance.Contrast(img).enhance(random.uniform(0.85, 1.15))
    img = ImageEnhance.Color(img).enhance(random.uniform(0.85, 1.15))
    
    # Slight blur
    blur_radius = random.uniform(0.3, 1.0)
    img = img.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    
    # Slight gaussian noise
    img_array = np.array(img).astype(np.float32)
    noise = np.random.normal(0, random.uniform(3, 12), img_array.shape)
    img_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)
    img = Image.fromarray(img_array)
    
    # Random small crop + resize back
    w, h = img.size
    crop_ratio = random.uniform(0.88, 0.96)
    crop_w = int(w * crop_ratio)
    crop_h = int(h * crop_ratio)
    left   = random.randint(0, w - crop_w)
    top    = random.randint(0, h - crop_h)
    img    = img.crop((left, top, left + crop_w, top + crop_h))
    img    = img.resize((w, h), Image.BILINEAR)
    
    return img


# =============================================================================
# ALL 5 AUGMENTATIONS TOGETHER
# =============================================================================

AUGMENTATIONS = [
    ("flip",       "Horizontal Flip",           augment_1_horizontal_flip),
    ("rot_pos",    "Rotation +20°",             augment_2_rotate_positive),
    ("rot_neg",    "Rotation -20°",             augment_3_rotate_negative),
    ("brightness", "Brightness & Contrast",     augment_4_brightness_contrast),
    ("combo",      "Combination Transform",     augment_5_combination),
]


# =============================================================================
# PREVIEW GENERATOR
# =============================================================================

def generate_preview():
    """Generate visual preview of all 5 augmentations"""
    
    print("\n[STEP 1] Generating augmentation preview...")
    
    # Find sample images
    samples = []
    for label in ['awake', 'sleepy']:
        src_dir = Path(Config.INPUT_DIR) / 'train' / label
        files   = list(src_dir.glob('*.jpg'))[:2]
        samples.extend([(f, label) for f in files])
    
    if not samples:
        print("No sample images found!")
        return
    
    # Total columns = original + 5 augmentations
    num_cols = 1 + len(AUGMENTATIONS)
    num_rows = len(samples)
    
    fig, axes = plt.subplots(
        num_rows, num_cols,
        figsize=(3 * num_cols, 3 * num_rows)
    )
    
    # Column headers
    col_names = ['Original'] + [aug[1] for aug in AUGMENTATIONS]
    
    for row, (img_path, label) in enumerate(samples):
        
        # Load and resize original
        img = Image.open(img_path).convert('RGB')
        img = img.resize(Config.IMAGE_SIZE, Image.BILINEAR)
        
        # Generate all augmentations
        augmented_images = [img] + [
            aug_func(img) for _, _, aug_func in AUGMENTATIONS
        ]
        
        for col, (aug_img, col_name) in enumerate(
                zip(augmented_images, col_names)):
            
            ax = axes[row, col] if num_rows > 1 else axes[col]
            ax.imshow(aug_img)
            ax.axis('off')
            
            # Column headers (top row only)
            if row == 0:
                ax.set_title(col_name, fontsize=9,
                            fontweight='bold', pad=5)
            
            # Row labels (first column only)
            if col == 0:
                ax.set_ylabel(
                    f"{label.upper()}\n{img_path.name[:15]}",
                    fontsize=8,
                    rotation=0,
                    labelpad=60,
                    va='center'
                )
    
    plt.suptitle(
        'MRL Dataset - 5 Augmentation Techniques Preview',
        fontsize=14,
        fontweight='bold',
        y=1.02
    )
    
    plt.tight_layout()
    
    os.makedirs('figures', exist_ok=True)
    save_path = 'figures/augmentation_preview.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"   Preview saved: {save_path}")


# =============================================================================
# DATASET AUGMENTOR
# =============================================================================

class DatasetAugmentor:
    
    def __init__(self):
        self.stats = defaultdict(lambda: defaultdict(int))
    
    def run(self):
        """Main augmentation pipeline"""
        
        print("\n" + "=" * 60)
        print("MRL DATASET AUGMENTATION")
        print("=" * 60)
        print(f"\nInput:    {Config.INPUT_DIR}")
        print(f"Output:   {Config.OUTPUT_DIR}")
        print(f"\nAugmentation Parts:")
        for i, (key, name, _) in enumerate(AUGMENTATIONS):
            print(f"  Part {i+1}: {name}")
        
        # Create output folder structure
        for split in ['train', 'val', 'test']:
            for label in ['awake', 'sleepy']:
                path = Path(Config.OUTPUT_DIR) / split / label
                path.mkdir(parents=True, exist_ok=True)
        
        print("\n" + "-" * 60)
        
        # Copy val and test unchanged
        for split in Config.COPY_SPLITS:
            print(f"\nCopying {split} (no augmentation)...")
            self._copy_split(split)
        
        # Augment training set
        for split in Config.AUGMENT_SPLITS:
            print(f"\nAugmenting {split} split...")
            self._augment_split(split)
        
        self._print_summary()
    
    def _copy_split(self, split):
        """Copy split without augmentation"""
        
        for label in ['awake', 'sleepy']:
            src_dir  = Path(Config.INPUT_DIR)  / split / label
            dest_dir = Path(Config.OUTPUT_DIR) / split / label
            
            if not src_dir.exists():
                print(f"  WARNING: {src_dir} not found")
                continue
            
            files = (list(src_dir.glob('*.jpg')) +
                     list(src_dir.glob('*.png')) +
                     list(src_dir.glob('*.jpeg')))
            
            for f in files:
                shutil.copy2(f, dest_dir / f.name)
                self.stats[split][label] += 1
            
            print(f"  {label}: {len(files):,} images copied")
    
    def _augment_split(self, split):
        """
        Augment split:
        Each image → 1 original + 5 augmented = 6 total
        """
        
        for label in ['awake', 'sleepy']:
            src_dir  = Path(Config.INPUT_DIR)  / split / label
            dest_dir = Path(Config.OUTPUT_DIR) / split / label
            
            if not src_dir.exists():
                print(f"  WARNING: {src_dir} not found")
                continue
            
            files = (list(src_dir.glob('*.jpg')) +
                     list(src_dir.glob('*.png')) +
                     list(src_dir.glob('*.jpeg')))
            
            print(f"\n  Processing {label} ({len(files):,} images)...")
            print(f"  Each image → 1 original + 5 augmented = 6 total")
            
            original_count  = 0
            augmented_count = 0
            failed_count    = 0
            
            for idx, src_file in enumerate(files):
                
                # ── Copy original ──
                dest_orig = dest_dir / f"orig_{src_file.name}"
                shutil.copy2(src_file, dest_orig)
                original_count += 1
                
                # ── Load image ──
                try:
                    img = Image.open(src_file).convert('RGB')
                    img = img.resize(Config.IMAGE_SIZE, Image.BILINEAR)
                except Exception as e:
                    print(f"    Warning: Could not open {src_file.name}: {e}")
                    failed_count += 1
                    continue
                
                # ── Apply all 5 augmentations ──
                for aug_key, aug_name, aug_func in AUGMENTATIONS:
                    try:
                        aug_img = aug_func(img)
                        
                        # Save with descriptive name
                        aug_filename = f"{aug_key}_{src_file.stem}.jpg"
                        aug_path     = dest_dir / aug_filename
                        aug_img.save(aug_path, 'JPEG', quality=95)
                        augmented_count += 1
                        
                    except Exception as e:
                        print(f"    Warning: {aug_name} failed "
                              f"on {src_file.name}: {e}")
                        failed_count += 1
                
                # Progress every 2000 images
                if (idx + 1) % 2000 == 0:
                    print(f"    [{idx+1:,}/{len(files):,}] "
                          f"Original: {original_count:,} | "
                          f"Augmented: {augmented_count:,}")
            
            total = original_count + augmented_count
            self.stats[split][label] = total
            
            print(f"\n  {label} Results:")
            print(f"    Original:   {original_count:,}")
            print(f"    Augmented:  {augmented_count:,} "
                  f"({len(AUGMENTATIONS)} per image)")
            print(f"    Failed:     {failed_count:,}")
            print(f"    Total:      {total:,}")
    
    def _print_summary(self):
        """Print final summary"""
        
        print("\n" + "=" * 60)
        print("AUGMENTATION COMPLETE - SUMMARY")
        print("=" * 60)
        
        # Original counts
        orig_train_awake  = len(list(
            (Path(Config.INPUT_DIR) / 'train' / 'awake').glob('*.jpg')))
        orig_train_sleepy = len(list(
            (Path(Config.INPUT_DIR) / 'train' / 'sleepy').glob('*.jpg')))
        
        grand_total = 0
        
        for split in ['train', 'val', 'test']:
            awake  = self.stats[split]['awake']
            sleepy = self.stats[split]['sleepy']
            total  = awake + sleepy
            grand_total += total
            
            print(f"\n{split.upper()}:")
            print(f"  Awake:  {awake:,}")
            print(f"  Sleepy: {sleepy:,}")
            print(f"  Total:  {total:,}")
            
            if total > 0:
                balance = (awake / total) * 100
                print(f"  Balance: {balance:.1f}% awake / "
                      f"{100-balance:.1f}% sleepy")
        
        print(f"\nGRAND TOTAL: {grand_total:,} images")
        
        print(f"\nMultiplier:")
        print(f"  Original train: "
              f"{orig_train_awake + orig_train_sleepy:,}")
        print(f"  Augmented train: "
              f"{self.stats['train']['awake'] + self.stats['train']['sleepy']:,}")
        
        aug_total = (self.stats['train']['awake'] +
                     self.stats['train']['sleepy'])
        orig_total = orig_train_awake + orig_train_sleepy
        
        if orig_total > 0:
            multiplier = aug_total / orig_total
            print(f"  Dataset size: {multiplier:.1f}x larger")
        
        print(f"\nOutput directory: {Config.OUTPUT_DIR}")
        print(f"\nTo train on augmented data:")
        print(f"  Set DATA_DIR = '{Config.OUTPUT_DIR}' in train_combined.py")


# =============================================================================
# VERIFY
# =============================================================================

def verify():
    """Quick verification of output"""
    
    print("\n" + "-" * 60)
    print("VERIFICATION")
    print("-" * 60)
    
    for split in ['train', 'val', 'test']:
        for label in ['awake', 'sleepy']:
            path = Path(Config.OUTPUT_DIR) / split / label
            
            if not path.exists():
                print(f"MISSING: {path}")
                continue
            
            all_files = (list(path.glob('*.jpg')) +
                         list(path.glob('*.png')))
            
            orig_files = [f for f in all_files
                          if f.name.startswith('orig_')]
            
            aug_files = {
                aug_key: [f for f in all_files
                          if f.name.startswith(f'{aug_key}_')]
                for aug_key, _, _ in AUGMENTATIONS
            }
            
            print(f"\n{split}/{label}:")
            print(f"  Original:  {len(orig_files):,}")
            for aug_key, aug_name, _ in AUGMENTATIONS:
                print(f"  {aug_name}: "
                      f"{len(aug_files[aug_key]):,}")
            print(f"  TOTAL:     {len(all_files):,}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    
    print("=" * 60)
    print("MRL EYE DATASET - 5 PART AUGMENTATION")
    print("=" * 60)
    
    # Step 1: Preview
    generate_preview()
    
    # Step 2: Augment
    print("\n[STEP 2] Running augmentation...")
    augmentor = DatasetAugmentor()
    augmentor.run()
    
    # Step 3: Verify
    print("\n[STEP 3] Verifying output...")
    verify()
    
    print("\n" + "=" * 60)
    print("ALL DONE!")
    print("=" * 60)
    print(f"\nOutput: {Config.OUTPUT_DIR}")
    print(f"Preview: figures/augmentation_preview.png")


if __name__ == "__main__":
    main()