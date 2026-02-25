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
import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from PIL import Image

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
    from yolo_benchmark import collect_labeled_images, load_ground_truth, calculate_iou
    
    pairs = collect_labeled_images(images_dir, labels_dir)
    samples = []

    for img_path, txt_path in pairs:
        gt_boxes = load_ground_truth(txt_path)
        pred_boxes = yolo_model.predict(img_path, conf_threshold=0.10)

        for pred in pred_boxes:
            cid = pred['class_id']
            if cid not in target_classes:
                continue

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

def ensure_models_trained(config, images_dir, labels_dir, yolo_model, target_classes):
    """Auto-training hook. Trains the models if they don't exist."""
    model_dir = Path(config.model_dir)
    os.makedirs(model_dir, exist_ok=True)
    
    all_exist = True
    for vc in VerifierClass:
        if not (model_dir / f"verifier_{vc.name.lower()}.pt").exists():
            all_exist = False
            break
            
    if all_exist:
        return

    print("\n[Verifier] Models missing. Starting auto-training...")
    print("=== Bronze: Collecting training crops ===")
    raw_samples = collect_training_data(images_dir, labels_dir, yolo_model, target_classes)
    
    print(f"\n=== Silver: Balancing dataset ({len(raw_samples)} raw samples) ===")
    balanced = balance_samples(raw_samples)

    print("\n=== Gold: Training verifiers ===")
    for vc in VerifierClass:
        train_verifier(config, balanced, vc)
        
    print("[Verifier] Auto-training complete.\n")

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
    def __init__(self, config: VerifierConfig):
        self.config = config
        self.models = {} 
        self._load_models()

    def _load_models(self):
        model_dir = Path(self.config.model_dir)
        for vc in VerifierClass:
            model_path = model_dir / f"verifier_{vc.name.lower()}.pt"
            if model_path.exists():
                model = build_verifier_model(self.config.model_name)
                model.load_state_dict(torch.load(model_path, map_location='cpu'))
                model.eval()
                self.models[vc] = model
                print(f"  [Verifier] Loaded {vc.name} verifier")

    def generate_votes(self, image_path, detections):
        """
        Evaluates YOLO detections and returns verifier predictions
        to be used as additional votes in Weighted Box Fusion.
        """
        if not self.models:
            return []

        img = Image.open(image_path).convert('RGB')
        img_w, img_h = img.size

        transform = T.Compose([
            T.Resize((self.config.input_size, self.config.input_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        votes = []

        for det in detections:
            cid = det['class_id']

            if cid in (vc.value for vc in VerifierClass):
                verifier_class = VerifierClass(cid)
            elif cid in CAPTION_TO_PARENT:
                verifier_class = CAPTION_TO_PARENT[cid]
            else:
                continue

            if verifier_class not in self.models:
                continue

            x1, y1, x2, y2 = det['bbox']
            x1_px = max(0, int(x1 * img_w))
            y1_px = max(0, int(y1 * img_h))
            x2_px = min(img_w, int(x2 * img_w))
            y2_px = min(img_h, int(y2 * img_h))
            
            if x2_px <= x1_px or y2_px <= y1_px:
                continue

            crop = img.crop((x1_px, y1_px, x2_px, y2_px))
            tensor = transform(crop).unsqueeze(0)
            
            with torch.no_grad():
                logit = self.models[verifier_class](tensor)
                prob = torch.sigmoid(logit).item()

            votes.append({
                'class_id': cid,
                'bbox': det['bbox'],
                'conf': prob
            })

        return votes


# =============================================================================
# TRAINING (run standalone)
# =============================================================================

def train_verifier(config, samples, verifier_class):
    class CropDataset(Dataset):
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
            
            x1_px = max(0, int(x1 * w))
            y1_px = max(0, int(y1 * h))
            x2_px = min(w, int(x2 * w))
            y2_px = min(h, int(y2 * h))
            
            if x2_px <= x1_px: x2_px = x1_px + 1
            if y2_px <= y1_px: y2_px = y1_px + 1
            
            crop = img.crop((x1_px, y1_px, x2_px, y2_px))
            tensor = self.transform(crop)
            label = torch.tensor([1.0 if s.is_positive else 0.0])
            return tensor, label

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

    for param in model.parameters():
        param.requires_grad = False
    for param in list(model.parameters())[-4:]:
        param.requires_grad = True

    for epoch in range(config.epochs):
        if epoch == config.freeze_backbone_epochs:
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