"""
crop_verifier.py - Second-stage CNN classifier for YOLO crop verification.

Architecture: YOLO proposes regions → crops are extracted → this CNN classifies
each crop as valid/invalid for its predicted class. This suppresses false
positives without hurting recall (YOLO stays sensitive, CNN filters noise).

Medallion architecture:
    Bronze: Raw crops from YOLO detections + ground truth labels
    Silver: Balanced training set with hard negatives
    Gold:   Trained verifier model ready for inference

Usage:
    # Training: build dataset from your existing labels + YOLO predictions
    python crop_verifier.py train --images-dir docs_images --labels-dir docs_labels

    # Inference: integrated via CropVerifier.verify() in the benchmark/pipeline
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum

# These imports are deferred to avoid hard dependency when only stubbing
# import torch
# import torch.nn as nn
# import torchvision.transforms as T
# import torchvision.models as models
# from torch.utils.data import Dataset, DataLoader
# from PIL import Image


# =============================================================================
# CONFIGURATION
# =============================================================================

class VerifierClass(Enum):
    """The classes we verify. Maps to TARGET_CLASSES in yolo_benchmark."""
    FIGURE = 3
    TABLE = 5
    FORMULA = 8


# Group captions with their parent for verification purposes
CAPTION_TO_PARENT = {
    4: VerifierClass.FIGURE,   # figure_caption → verify as figure-adjacent
    6: VerifierClass.TABLE,    # table_caption → verify as table-adjacent
    9: VerifierClass.FORMULA,  # formula_caption → verify as formula-adjacent
    7: VerifierClass.TABLE,    # table_footnote → verify as table-adjacent
}


@dataclass
class VerifierConfig:
    model_name: str = "efficientnet_b0"  # Small, fast, good enough for binary
    input_size: int = 224
    batch_size: int = 32
    epochs: int = 15
    lr: float = 1e-4
    freeze_backbone_epochs: int = 5  # Freeze pretrained layers initially
    confidence_threshold: float = 0.5  # Reject crops below this
    model_dir: str = "verifier_models"
    # Per-class override thresholds (tune after training)
    class_thresholds: dict = field(default_factory=lambda: {
        VerifierClass.FIGURE: 0.5,
        VerifierClass.TABLE: 0.5,
        VerifierClass.FORMULA: 0.5,
    })


# =============================================================================
# BRONZE LAYER: Raw Data Collection
# =============================================================================

@dataclass
class CropSample:
    """A single training sample for the verifier."""
    image_path: str
    bbox_normalized: tuple  # (x1, y1, x2, y2) in [0,1]
    predicted_class: int
    is_positive: bool  # True if this crop matches a ground truth box


def collect_training_data(images_dir, labels_dir, yolo_model, target_classes):
    """
    Bronze layer: Generate training crops by running YOLO on labeled images
    and matching predictions to ground truth.

    Positives: YOLO detections that match a GT box (IoU >= 0.5)
    Negatives: YOLO detections that don't match any GT box (hard negatives)
               + random background crops (easy negatives)

    Returns list of CropSample.
    """
    from yolo_benchmark import (
        collect_labeled_images, load_ground_truth,
        get_predictions, calculate_iou
    )

    pairs = collect_labeled_images(images_dir, labels_dir)
    samples = []

    for img_path, txt_path in pairs:
        gt_boxes = load_ground_truth(txt_path)
        # Use low confidence to get more candidate detections
        pred_boxes = get_predictions(yolo_model, img_path, conf_threshold=0.10)

        for pred in pred_boxes:
            cid = pred['class_id']
            if cid not in target_classes:
                continue

            # Check if this prediction matches any GT box
            is_positive = False
            for gt in gt_boxes:
                if gt['class_id'] == cid:
                    iou = calculate_iou(pred['bbox'], gt['bbox'])
                    if iou >= 0.5:
                        is_positive = True
                        break

            samples.append(CropSample(
                image_path=img_path,
                bbox_normalized=pred['bbox'],
                predicted_class=cid,
                is_positive=is_positive,
            ))

    return samples


# =============================================================================
# SILVER LAYER: Balanced Dataset
# =============================================================================

def balance_samples(samples, max_ratio=3.0):
    """
    Silver layer: Balance positives and negatives per class.
    Caps negative samples at max_ratio * num_positives to prevent
    the model from just predicting "reject everything".
    """
    import random

    by_class = {}
    for s in samples:
        by_class.setdefault(s.predicted_class, {'pos': [], 'neg': []})
        if s.is_positive:
            by_class[s.predicted_class]['pos'].append(s)
        else:
            by_class[s.predicted_class]['neg'].append(s)

    balanced = []
    for cid, splits in by_class.items():
        positives = splits['pos']
        negatives = splits['neg']

        max_neg = int(len(positives) * max_ratio)
        if len(negatives) > max_neg:
            negatives = random.sample(negatives, max_neg)

        balanced.extend(positives)
        balanced.extend(negatives)

        print(f"  Class {cid}: {len(positives)} pos, {len(negatives)} neg")

    random.shuffle(balanced)
    return balanced


# =============================================================================
# GOLD LAYER: Model Definition
# =============================================================================

def build_verifier_model(model_name="efficientnet_b0", num_classes=1):
    """
    Build a binary classifier (sigmoid output) from a pretrained backbone.

    Using a single sigmoid output per verifier (one model per parent class)
    keeps things simple and lets you tune thresholds independently.
    """
    import torch.nn as nn
    import torchvision.models as models

    if model_name == "efficientnet_b0":
        backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        num_features = backbone.classifier[1].in_features
        backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(num_features, num_classes),
        )
    elif model_name == "resnet18":
        backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        num_features = backbone.fc.in_features
        backbone.fc = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(num_features, num_classes),
        )
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    return backbone


# =============================================================================
# INFERENCE INTEGRATION
# =============================================================================

class CropVerifier:
    """
    Integrates with EnsembleYOLO to filter false positives.

    Usage in benchmark/pipeline:
        ensemble = EnsembleYOLO(config)
        verifier = CropVerifier(verifier_config)

        detections = ensemble.predict(image_path)
        verified = verifier.verify(image_path, detections)
    """

    def __init__(self, config: VerifierConfig):
        self.config = config
        self.models = {}  # VerifierClass -> loaded model
        self._load_models()

    def _load_models(self):
        """Load trained verifier models from disk."""
        import torch

        model_dir = Path(self.config.model_dir)
        if not model_dir.exists():
            print(f"  [Verifier] No model dir at {model_dir}, running without verification")
            return

        for vc in VerifierClass:
            model_path = model_dir / f"verifier_{vc.name.lower()}.pt"
            if model_path.exists():
                model = build_verifier_model(self.config.model_name)
                model.load_state_dict(torch.load(model_path, map_location='cpu'))
                model.eval()
                self.models[vc] = model
                print(f"  [Verifier] Loaded {vc.name} verifier")

    def verify(self, image_path, detections):
        """
        Filter detections by running each crop through the appropriate
        verifier model. Returns only detections that pass.
        """
        if not self.models:
            return detections  # No verifiers loaded, pass everything through

        import torch
        from PIL import Image
        import torchvision.transforms as T

        img = Image.open(image_path).convert('RGB')
        img_w, img_h = img.size

        transform = T.Compose([
            T.Resize((self.config.input_size, self.config.input_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        verified = []

        for det in detections:
            cid = det['class_id']

            # Determine which verifier to use
            if cid in (vc.value for vc in VerifierClass):
                verifier_class = VerifierClass(cid)
            elif cid in CAPTION_TO_PARENT:
                verifier_class = CAPTION_TO_PARENT[cid]
            else:
                verified.append(det)  # No verifier for this class, keep it
                continue

            if verifier_class not in self.models:
                verified.append(det)
                continue

            # Extract crop
            x1, y1, x2, y2 = det['bbox']
            crop = img.crop((
                int(x1 * img_w), int(y1 * img_h),
                int(x2 * img_w), int(y2 * img_h),
            ))

            # Classify
            tensor = transform(crop).unsqueeze(0)
            with torch.no_grad():
                logit = self.models[verifier_class](tensor)
                prob = torch.sigmoid(logit).item()

            threshold = self.config.class_thresholds.get(
                verifier_class, self.config.confidence_threshold
            )

            if prob >= threshold:
                verified.append(det)

        return verified


# =============================================================================
# TRAINING (run standalone)
# =============================================================================

def train_verifier(config, samples, verifier_class):
    """
    Train a single binary verifier for one parent class.
    Implements the two-phase strategy: frozen backbone → full fine-tune.
    """
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    from PIL import Image
    import torchvision.transforms as T

    class CropDataset:
        def __init__(self, samples, transform):
            self.samples = samples
            self.transform = transform

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            s = self.samples[idx]
            img = Image.open(s.image_path).convert('RGB')
            w, h = img.size
            x1, y1, x2, y2 = s.bbox_normalized
            crop = img.crop((int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h)))
            tensor = self.transform(crop)
            label = torch.tensor([1.0 if s.is_positive else 0.0])
            return tensor, label

    # Filter samples for this verifier class
    relevant_cids = {verifier_class.value}
    for caption_cid, parent in CAPTION_TO_PARENT.items():
        if parent == verifier_class:
            relevant_cids.add(caption_cid)

    class_samples = [s for s in samples if s.predicted_class in relevant_cids]
    if not class_samples:
        print(f"  No samples for {verifier_class.name}, skipping")
        return

    print(f"\nTraining {verifier_class.name} verifier on {len(class_samples)} samples")

    transform = T.Compose([
        T.Resize((config.input_size, config.input_size)),
        T.RandomHorizontalFlip(p=0.3),
        T.ColorJitter(brightness=0.2, contrast=0.2),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset = CropDataset(class_samples, transform)
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    model = build_verifier_model(config.model_name)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    # Phase 1: Frozen backbone
    for param in model.parameters():
        param.requires_grad = False
    # Unfreeze classifier head
    for param in list(model.parameters())[-4:]:
        param.requires_grad = True

    for epoch in range(config.epochs):
        if epoch == config.freeze_backbone_epochs:
            # Phase 2: Unfreeze everything with lower LR
            for param in model.parameters():
                param.requires_grad = True
            optimizer = torch.optim.Adam(model.parameters(), lr=config.lr / 10)

        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for inputs, labels in loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = (torch.sigmoid(outputs) >= 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        acc = correct / total if total > 0 else 0
        print(f"  Epoch {epoch+1}/{config.epochs} - Loss: {total_loss/len(loader):.4f}, Acc: {acc:.3f}")

    # Save
    os.makedirs(config.model_dir, exist_ok=True)
    save_path = os.path.join(config.model_dir, f"verifier_{verifier_class.name.lower()}.pt")
    torch.save(model.state_dict(), save_path)
    print(f"  Saved: {save_path}")


# =============================================================================
# CLI
# =============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Train crop verifier models")
    parser.add_argument('action', choices=['train', 'evaluate'])
    parser.add_argument('--images-dir', default='docs_images')
    parser.add_argument('--labels-dir', default='docs_labels')
    parser.add_argument('--model-path', default='doclayout_yolo_docstructbench_imgsz1024.pt')
    args = parser.parse_args()

    from yolo_benchmark import SingleScaleYOLO, TARGET_CLASSES

    config = VerifierConfig()

    if args.action == 'train':
        print("=== Bronze: Collecting training crops ===")
        yolo = SingleScaleYOLO(args.model_path)
        raw_samples = collect_training_data(
            args.images_dir, args.labels_dir, yolo, TARGET_CLASSES
        )
        print(f"  Total raw samples: {len(raw_samples)}")

        print("\n=== Silver: Balancing dataset ===")
        balanced = balance_samples(raw_samples)

        print("\n=== Gold: Training verifiers ===")
        for vc in VerifierClass:
            train_verifier(config, balanced, vc)

        print("\nDone! Models saved to:", config.model_dir)


if __name__ == '__main__':
    main()