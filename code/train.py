"""
Combined MRL + NTHU-DDD Training Script
Train on merged dataset for maximum robustness
Generates BOTH Training and Validation curves
"""
import matplotlib
matplotlib.use('Agg')  # Fix Tcl/Tk error on Windows

import os
import sys
import time
import logging
from datetime import datetime

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm

from datasets import load_dataset
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from torchvision.transforms import (
    Compose, Resize, ToTensor, Normalize,
    RandomHorizontalFlip, RandomRotation, ColorJitter,
    RandomAffine, GaussianBlur, RandomErasing
)
from torch.utils.data import DataLoader
import evaluate
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# =============================================================================
# CONFIGURATION
# =============================================================================

class Config:
    """Training configuration for MRL dataset - ViT-Base"""
    
    # Dataset
    DATA_DIR = "data/MRL_AUGMENTED"
    
    # Model - ViT Base
    MODEL_NAME = "google/vit-base-patch16-224"
    
    OUTPUT_DIR = "./models/vit-base-mrl-augmented"
    
    # Training hyperparameters
    BATCH_SIZE = 8
    LEARNING_RATE = 2e-5
    NUM_EPOCHS = 15
    WARMUP_RATIO = 0.1
    WEIGHT_DECAY = 0.01
    
    # Early stopping
    EARLY_STOPPING_PATIENCE = 3
    
    # Data augmentation strength
    AUGMENTATION_STRENGTH = "strong"
    
    # Subset for testing
    USE_SUBSET = False
    SUBSET_SIZE = 5000
    
    # Advanced
    GRADIENT_ACCUMULATION_STEPS = 4
    FP16_TRAINING = True
    LABEL_SMOOTHING = 0.1
    
    # Random seed
    SEED = 42

# =============================================================================
# LOGGING SETUP
# =============================================================================

def setup_logging():
    """Setup logging to both file and console"""
    os.makedirs("logs", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"logs/training_combined_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Log file: {log_file}")
    return logger

# =============================================================================
# DATA AUGMENTATION
# =============================================================================

def get_augmentation_transforms(image_processor, strength="medium"):
    base_mean = image_processor.image_mean
    base_std = image_processor.image_std
    
    if strength == "weak":
        train_transform = Compose([
            Resize((224, 224)),
            RandomHorizontalFlip(p=0.5),
            ToTensor(),
            Normalize(mean=base_mean, std=base_std)
        ])
    elif strength == "medium":
        train_transform = Compose([
            Resize((224, 224)),
            RandomHorizontalFlip(p=0.5),
            RandomRotation(degrees=10),
            ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            ToTensor(),
            Normalize(mean=base_mean, std=base_std)
        ])
    elif strength == "strong":
        train_transform = Compose([
            Resize((224, 224)),
            RandomHorizontalFlip(p=0.5),
            RandomRotation(degrees=15),
            ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
            RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
            ToTensor(),
            Normalize(mean=base_mean, std=base_std),
            RandomErasing(p=0.2, scale=(0.02, 0.15))
        ])
    else:
        raise ValueError(f"Unknown augmentation strength: {strength}")
    
    val_transform = Compose([
        Resize((224, 224)),
        ToTensor(),
        Normalize(mean=base_mean, std=base_std)
    ])
    
    return train_transform, val_transform

# =============================================================================
# METRICS
# =============================================================================

def create_compute_metrics():
    accuracy_metric = evaluate.load("accuracy")
    precision_metric = evaluate.load("precision")
    recall_metric = evaluate.load("recall")
    f1_metric = evaluate.load("f1")
    
    def compute_metrics(eval_pred):
        predictions = np.argmax(eval_pred.predictions, axis=1)
        labels = eval_pred.label_ids
        
        return {
            "accuracy": accuracy_metric.compute(predictions=predictions, references=labels)["accuracy"],
            "precision": precision_metric.compute(predictions=predictions, references=labels, average="weighted")["precision"],
            "recall": recall_metric.compute(predictions=predictions, references=labels, average="weighted")["recall"],
            "f1": f1_metric.compute(predictions=predictions, references=labels, average="weighted")["f1"]
        }
    
    return compute_metrics

# =============================================================================
# NEW: Calculate Train Accuracy from Checkpoints
# =============================================================================

def calculate_train_accuracy_per_epoch(trainer, dataset, epochs_count, device):
    """
    Runs inference on training set for each saved checkpoint to get 
    accurate Training Accuracy per epoch.
    """
    print("\n" + "="*70)
    print("CALCULATING TRAINING ACCURACY PER EPOCH")
    print("="*70)
    
    train_acc_history = []
    
    # We assume checkpoints are saved every epoch due to save_strategy="epoch"
    # We will load them sequentially
    output_dir = trainer.args.output_dir
    
    # Find all epoch checkpoints
    checkpoints = []
    for i in range(1, epochs_count + 1):
        # Checkpoint naming convention: checkpoint-<steps>
        # We need to map epoch to steps roughly or just look for folders
        # Since we save every epoch, let's look for folders containing 'checkpoint'
        pass
    
    # Better approach: Scan directory for checkpoint folders
    all_dirs = [d for d in os.listdir(output_dir) if d.startswith("checkpoint-")]
    all_dirs.sort(key=lambda x: int(x.split('-')[1])) # Sort by step number
    
    # Limit to number of epochs
    if len(all_dirs) > epochs_count:
        all_dirs = all_dirs[:epochs_count]
        
    print(f"Found {len(all_dirs)} checkpoints corresponding to epochs.")
    
    # Prepare a small subset of training data for speed (optional, but recommended)
    # Or use full train set if you want exact numbers (slower)
    train_subset = dataset.select(range(min(5000, len(dataset)))) # Use 5k samples for speed
    print(f"Evaluating on subset of {len(train_subset)} training samples per epoch...")
    
    # Create a simple dataloader for this subset
    # Note: We reuse the transforms already applied to the dataset
    collate_fn = lambda batch: {
        "pixel_values": torch.stack([b["pixel_values"] for b in batch]),
        "labels": torch.tensor([b["label"] for b in batch])
    }
    
    loader = DataLoader(train_subset, batch_size=32, shuffle=False, collate_fn=collate_fn)
    
    for i, ckpt_dir_name in enumerate(all_dirs):
        ckpt_path = os.path.join(output_dir, ckpt_dir_name)
        print(f"\nEpoch {i+1}: Evaluating checkpoint {ckpt_dir_name}...")
        
        try:
            # Load model from this specific checkpoint
            temp_model = AutoModelForImageClassification.from_pretrained(ckpt_path)
            temp_model.to(device)
            temp_model.eval()
            
            correct = 0
            total = 0
            
            with torch.no_grad():
                for batch in tqdm(loader, desc=f"  Epoch {i+1} Eval"):
                    inputs = {k: v.to(device) for k, v in batch.items()}
                    outputs = temp_model(**inputs)
                    preds = outputs.logits.argmax(dim=-1)
                    
                    correct += (preds == inputs["labels"]).sum().item()
                    total += inputs["labels"].size(0)
            
            acc = correct / total
            train_acc_history.append(acc)
            print(f"  -> Train Acc: {acc*100:.2f}%")
            
            # Clean up memory
            del temp_model
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"  -> Error evaluating checkpoint: {e}")
            train_acc_history.append(0.0) # Fallback
            
    return train_acc_history

# =============================================================================
# VISUALIZATION (UPDATED)
# =============================================================================

def plot_confusion_matrix(trainer, dataset, id2label, split_name, save_dir):
    predictions = trainer.predict(dataset)
    y_pred = np.argmax(predictions.predictions, axis=1)
    y_true = predictions.label_ids
    
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=[id2label[0], id2label[1]],
                yticklabels=[id2label[0], id2label[1]])
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('Actual', fontsize=12)
    plt.title(f'Confusion Matrix - {split_name}', fontsize=14)
    
    save_path = os.path.join(save_dir, f'confusion_matrix_{split_name}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    return save_path

def plot_training_curves_full(log_history, train_acc_history, save_dir):
    """Plot BOTH Training and Validation curves"""
    
    eval_loss = []
    eval_acc = []
    epochs_list = []
    
    for entry in log_history:
        if 'eval_loss' in entry:
            eval_loss.append(entry['eval_loss'])
            eval_acc.append(entry.get('eval_accuracy', 0))
            epochs_list.append(int(entry['epoch']))
    
    if not epochs_list:
        return None
    
    # Ensure train_acc_history matches length
    if len(train_acc_history) != len(epochs_list):
        print(f"Warning: Train acc history length ({len(train_acc_history)}) != Val epochs ({len(epochs_list)})")
        # Trim or pad if necessary, but usually they match if saved every epoch
        min_len = min(len(train_acc_history), len(epochs_list))
        train_acc_history = train_acc_history[:min_len]
        eval_acc = eval_acc[:min_len]
        eval_loss = eval_loss[:min_len]
        epochs_list = epochs_list[:min_len]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # --- Plot 1: Accuracy (Train vs Val) ---
    ax1.plot(epochs_list, [a * 100 for a in eval_acc], 'g-o', linewidth=2, markersize=8, label='Validation Acc')
    if train_acc_history:
        ax1.plot(epochs_list, [a * 100 for a in train_acc_history], 'b-s', linewidth=2, markersize=8, label='Training Acc')
    
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Accuracy (%)', fontsize=12)
    ax1.set_title('Training vs Validation Accuracy', fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # --- Plot 2: Loss (Train Log vs Val) ---
    # Extract training loss (logged frequently, we take the last one per epoch approx)
    train_loss_per_epoch = []
    current_epoch = 0
    current_loss_sum = 0
    count = 0
    
    # Simple approximation: take the last 'loss' entry before each 'eval_loss' entry
    # Or just plot all train losses as a line (noisy) vs smooth val loss
    # Let's do the smooth version:
    
    # Re-scan log_history to group train losses by epoch
    epoch_train_losses = {}
    for entry in log_history:
        if 'loss' in entry and 'eval_loss' not in entry:
            ep = int(entry.get('epoch', 0))
            if ep not in epoch_train_losses:
                epoch_train_losses[ep] = []
            epoch_train_losses[ep].append(entry['loss'])
            
    smoothed_train_loss = []
    for ep in epochs_list:
        if ep in epoch_train_losses:
            smoothed_train_loss.append(np.mean(epoch_train_losses[ep]))
        else:
            smoothed_train_loss.append(None)
            
    # Filter out Nones for plotting if any
    valid_indices = [i for i, x in enumerate(smoothed_train_loss) if x is not None]
    plot_epochs = [epochs_list[i] for i in valid_indices]
    plot_train_loss = [smoothed_train_loss[i] for i in valid_indices]
    plot_val_loss = [eval_loss[i] for i in valid_indices]
    
    if plot_train_loss:
        ax2.plot(plot_epochs, plot_train_loss, 'b-s', linewidth=2, markersize=8, label='Training Loss')
    ax2.plot(plot_epochs, plot_val_loss, 'g-o', linewidth=2, markersize=8, label='Validation Loss')
    
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Loss', fontsize=12)
    ax2.set_title('Training vs Validation Loss', fontsize=14)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'training_curves_full.png')
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    return save_path

# =============================================================================
# MAIN TRAINING PIPELINE
# =============================================================================

def main():
    print("\n" + "=" * 70)
    print("  COMBINED DATASET TRAINING (MRL + NTHU-DDD)")
    print("  Vision Transformer for Drowsiness Detection")
    print("=" * 70)
    
    logger = setup_logging()
    
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"\nGPU: {gpu_name}")
        logger.info(f"GPU Memory: {gpu_mem:.1f} GB")
        logger.info(f"CUDA Version: {torch.version.cuda}")
        torch.cuda.empty_cache()
    else:
        logger.warning("\nNo GPU detected!")
    
    logger.info("\n" + "-" * 70)
    logger.info("CONFIGURATION")
    logger.info("-" * 70)
    for key, value in vars(Config).items():
        if not key.startswith('_'):
            logger.info(f"  {key}: {value}")
    
    torch.manual_seed(Config.SEED)
    np.random.seed(Config.SEED)
    
    # =========================================================================
    # STEP 1: Load Dataset
    # =========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("STEP 1: LOADING DATASET")
    logger.info("=" * 70)
    
    if not os.path.exists(Config.DATA_DIR):
        logger.error(f"\nDataset not found: {Config.DATA_DIR}")
        sys.exit(1)
    
    dataset = load_dataset("imagefolder", data_dir=Config.DATA_DIR)
    
    val_split = None
    for name in ["validation", "val", "valid"]:
        if name in dataset:
            val_split = name
            break
    
    if val_split is None:
        logger.info("No validation split found. Creating from training data...")
        split_data = dataset["train"].train_test_split(test_size=0.15, seed=Config.SEED)
        dataset["train"] = split_data["train"]
        dataset["validation"] = split_data["test"]
        val_split = "validation"
    
    if Config.USE_SUBSET:
        logger.info(f"\n⚠ Using SUBSET mode")
        dataset["train"] = dataset["train"].shuffle(seed=Config.SEED).select(
            range(min(Config.SUBSET_SIZE, len(dataset["train"])))
        )
        dataset[val_split] = dataset[val_split].shuffle(seed=Config.SEED).select(
            range(min(Config.SUBSET_SIZE // 5, len(dataset[val_split])))
        )
    
    logger.info(f"\nDataset sizes:")
    logger.info(f"  Train:      {len(dataset['train']):,} samples")
    logger.info(f"  Validation: {len(dataset[val_split]):,} samples")
    if "test" in dataset:
        logger.info(f"  Test:       {len(dataset['test']):,} samples")
    
    # =========================================================================
    # STEP 2: Setup Labels
    # =========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("STEP 2: LABEL CONFIGURATION")
    logger.info("=" * 70)
    
    labels = dataset["train"].features["label"].names
    label2id = {label: i for i, label in enumerate(labels)}
    id2label = {i: label for i, label in enumerate(labels)}
    
    logger.info(f"Classes: {labels}")
    
    # =========================================================================
    # STEP 3: Image Preprocessing
    # =========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("STEP 3: PREPROCESSING")
    logger.info("=" * 70)
    
    image_processor = AutoImageProcessor.from_pretrained(Config.MODEL_NAME)
    train_transform, val_transform = get_augmentation_transforms(
        image_processor, strength=Config.AUGMENTATION_STRENGTH
    )
    
    def apply_train_transforms(batch):
        images = [img.convert("RGB") for img in batch["image"]]
        batch["pixel_values"] = [train_transform(img) for img in images]
        return batch
    
    def apply_val_transforms(batch):
        images = [img.convert("RGB") for img in batch["image"]]
        batch["pixel_values"] = [val_transform(img) for img in images]
        return batch
    
    dataset["train"] = dataset["train"].with_transform(apply_train_transforms)
    dataset[val_split] = dataset[val_split].with_transform(apply_val_transforms)
    if "test" in dataset:
        dataset["test"] = dataset["test"].with_transform(apply_val_transforms)
    
    # =========================================================================
    # STEP 4: Load Model
    # =========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("STEP 4: LOADING MODEL")
    logger.info("=" * 70)
    
    model = AutoModelForImageClassification.from_pretrained(
        Config.MODEL_NAME,
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Total parameters: {total_params:,}")
    
    # =========================================================================
    # STEP 5: Training Configuration
    # =========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("STEP 5: CONFIGURATION")
    logger.info("=" * 70)
    
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    
    training_args = TrainingArguments(
        output_dir=Config.OUTPUT_DIR,
        num_train_epochs=Config.NUM_EPOCHS,
        per_device_train_batch_size=Config.BATCH_SIZE,
        per_device_eval_batch_size=Config.BATCH_SIZE,
        gradient_accumulation_steps=Config.GRADIENT_ACCUMULATION_STEPS,
        learning_rate=Config.LEARNING_RATE,
        warmup_ratio=Config.WARMUP_RATIO,
        weight_decay=Config.WEIGHT_DECAY,
        fp16=Config.FP16_TRAINING and torch.cuda.is_available(),
        label_smoothing_factor=Config.LABEL_SMOOTHING,
        eval_strategy="epoch",
        save_strategy="epoch", # Crucial for our accuracy calculation
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        logging_dir=os.path.join(Config.OUTPUT_DIR, "logs"),
        logging_steps=50,
        report_to="none",
        remove_unused_columns=False,
        seed=Config.SEED,
        dataloader_num_workers=0,
        dataloader_pin_memory=True,
    )
    
    # =========================================================================
    # STEP 6 & 7: Collator & Trainer
    # =========================================================================
    def collate_fn(batch):
        pixel_values = torch.stack([item["pixel_values"] for item in batch])
        labels = torch.tensor([item["label"] for item in batch])
        return {"pixel_values": pixel_values, "labels": labels}
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset[val_split],
        compute_metrics=create_compute_metrics(),
        data_collator=collate_fn,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=Config.EARLY_STOPPING_PATIENCE)]
    )
    
    # =========================================================================
    # STEP 8: START TRAINING
    # =========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("STEP 8: TRAINING STARTED")
    logger.info("=" * 70)
    
    start_time = time.time()
    logger.info(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        train_result = trainer.train()
        training_time = time.time() - start_time
        logger.info(f"\n✅ Training completed in {training_time/60:.1f} minutes")
    except KeyboardInterrupt:
        logger.warning("\n⚠ Interrupted!")
        trainer.save_model(os.path.join(Config.OUTPUT_DIR, "interrupted"))
        sys.exit(0)
    except Exception as e:
        logger.error(f"\n❌ Failed: {str(e)}")
        raise
    
    # =========================================================================
    # STEP 9: EVALUATION
    # =========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("STEP 9: FINAL EVALUATION")
    logger.info("=" * 70)
    
    val_results = trainer.evaluate(dataset[val_split])
    logger.info(f"Val Acc: {val_results['eval_accuracy']*100:.2f}%")
    
    test_results = None
    if "test" in dataset:
        test_results = trainer.evaluate(dataset["test"])
        logger.info(f"Test Acc: {test_results['eval_accuracy']*100:.2f}%")
    
    # =========================================================================
    # STEP 10: SAVE & GENERATE CURVES (WITH TRAIN ACC)
    # =========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("STEP 10: SAVING & PLOTTING")
    logger.info("=" * 70)
    
    final_model_path = os.path.join(Config.OUTPUT_DIR, "final_model")
    trainer.save_model(final_model_path)
    image_processor.save_pretrained(final_model_path)
    logger.info(f"✅ Model saved to: {final_model_path}")
    
    # Confusion Matrix
    try:
        cm_path = plot_confusion_matrix(trainer, dataset[val_split], id2label, "validation", Config.OUTPUT_DIR)
        logger.info(f"✅ Confusion matrix: {cm_path}")
    except Exception as e:
        logger.warning(f"CM Error: {e}")
    
    # *** NEW: Calculate Train Accuracy per Epoch ***
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_acc_history = calculate_train_accuracy_per_epoch(
        trainer, dataset["train"], Config.NUM_EPOCHS, device
    )
    
    # *** NEW: Plot Full Curves ***
    try:
        curves_path = plot_training_curves_full(
            trainer.state.log_history, 
            train_acc_history, 
            Config.OUTPUT_DIR
        )
        if curves_path:
            logger.info(f"✅ Full training curves saved: {curves_path}")
    except Exception as e:
        logger.warning(f"Curve Error: {e}")
    
    # =========================================================================
    # STEP 11: SUMMARY
    # =========================================================================
    summary = f"""
    TRAINING COMPLETE
    ────────────────────────────────────────
    Model: {Config.MODEL_NAME}
    Time: {training_time/60:.1f} mins
    
    Results:
      Val Acc: {val_results['eval_accuracy']*100:.2f}%
      Test Acc: {test_results['eval_accuracy']*100:.2f}% if test_results else "N/A"
    
    Output: {final_model_path}
    Curves: {Config.OUTPUT_DIR}/training_curves_full.png
    ────────────────────────────────────────
    Ready!
    """
    
    logger.info(summary)
    print(summary)

if __name__ == "__main__":
    main()