import os
import json
import torch
import torch.nn as nn
import psutil
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from torchvision import models

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)

from data.dataset_loader import get_dataloaders, train_transform, test_transform


# ==================================================
# FEATURE EXTRACTOR
# ==================================================

class V3LargeFeatureExtractor(nn.Module):
    """
    MobileNetV3 Large frozen hoàn toàn.
    Trích xuất feature vector 960 chiều tại
    Global Average Pooling — sau features, trước classifier.
    """

    def __init__(self):
        super().__init__()

        backbone = models.mobilenet_v3_large(
            weights=models.MobileNet_V3_Large_Weights.DEFAULT
        )

        # Chỉ lấy phần features + avgpool, bỏ classifier
        self.features = backbone.features
        self.avgpool = backbone.avgpool

        # Frozen hoàn toàn — không train gì cả
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)  # (B, 960)

        return x


# ==================================================
# ML TRAINER
# ==================================================

class MLTrainer:

    def __init__(
            self,
            classifier_name,
            device,
            save_dir,
            **classifier_kwargs
    ):
        """
        classifier_name : 'svm' | 'knn' | 'random_forest'
        classifier_kwargs: tham số truyền vào sklearn classifier
        """

        self.classifier_name = classifier_name
        self.device = device
        self.save_dir = save_dir

        os.makedirs(self.save_dir, exist_ok=True)

        # Khởi tạo feature extractor
        self.extractor = V3LargeFeatureExtractor().to(device)
        self.extractor.eval()

        # Khởi tạo classifier
        self.classifier = self._build_classifier(
            classifier_name,
            **classifier_kwargs
        )

        # Metrics
        self.training_time = 0
        self.cpu_ram_mb = 0

    def _build_classifier(
            self,
            name,
            **kwargs
    ):

        if name == "svm":
            return SVC(
                kernel=kwargs.get("kernel", "rbf"),
                C=kwargs.get("C", 1.0),
                probability=True
            )

        elif name == "knn":
            return KNeighborsClassifier(
                n_neighbors=kwargs.get("n_neighbors", 5)
            )

        elif name == "random_forest":
            return RandomForestClassifier(
                n_estimators=kwargs.get("n_estimators", 100),
                random_state=42
            )

        else:
            raise ValueError(
                f"classifier_name phải là 'svm', 'knn', hoặc 'random_forest'. "
                f"Nhận được: '{name}'"
            )

    # ==================================================
    # EXTRACT FEATURES
    # ==================================================

    def extract_features(self, loader, desc="Extracting"):
        """Trích xuất feature vectors từ toàn bộ loader."""

        all_features = []
        all_labels = []

        with torch.no_grad():
            for images, labels in tqdm(loader, desc=desc):
                images = images.to(self.device)
                features = self.extractor(images)

                all_features.append(
                    features.cpu().numpy()
                )

                all_labels.extend(
                    labels.numpy()
                )

        return (
            np.vstack(all_features),
            np.array(all_labels)
        )

    # ==================================================
    # FIT
    # ==================================================

    def fit(
            self,
            train_loader,
            val_loader
    ):

        print(f"\nTraining {self.classifier_name.upper()}...\n")

        # Trích xuất features train
        print("Extracting train features...")
        X_train, y_train = self.extract_features(
            train_loader,
            desc="Train"
        )

        # Trích xuất features val
        print("Extracting val features...")
        X_val, y_val = self.extract_features(
            val_loader,
            desc="Val"
        )

        # Train classifier
        self.classifier.fit(X_train, y_train)

        # Validate
        val_preds = self.classifier.predict(X_val)
        val_acc = accuracy_score(y_val, val_preds) * 100

        self.cpu_ram_mb = (
                psutil.Process().memory_info().rss / 1024 / 1024
        )

        print(f"\nTraining Finished")
        print(f"Val Accuracy  : {val_acc:.2f}%")
        print(f"CPU RAM       : {self.cpu_ram_mb:.2f} MB")

    # ==================================================
    # EVALUATE TEST
    # ==================================================

    def evaluate_test(
            self,
            test_loader,
            class_names=None
    ):

        print("\nExtracting test features...")

        all_features = []
        all_labels = []

        with torch.no_grad():
            for images, labels in tqdm(
                    test_loader,
                    desc="Test"
            ):
                images = images.to(self.device)
                features = self.extractor(images)

                all_features.append(features.cpu().numpy())
                all_labels.extend(labels.numpy())

        X_test = np.vstack(all_features)
        y_test = np.array(all_labels)

        preds = self.classifier.predict(X_test)

        # Metrics
        acc = accuracy_score(y_test, preds)

        precision = precision_score(
            y_test, preds, average="weighted"
        )

        recall = recall_score(
            y_test, preds, average="weighted"
        )

        f1 = f1_score(
            y_test, preds, average="weighted"
        )

        self._save_confusion_matrix(
            y_test, preds, class_names
        )

        print(f"\n========== TEST ({self.classifier_name.upper()}) ==========")
        print(f"Accuracy         : {acc * 100:.2f}%")
        print(f"Precision        : {precision:.4f}")
        print(f"Recall           : {recall:.4f}")
        print(f"F1-score         : {f1:.4f}")
        print("=" * 50 + "\n")

        with open(
                os.path.join(self.save_dir, "test_metrics.json"),
                "w"
        ) as f:
            json.dump(
                {
                    "classifier": self.classifier_name,
                    "accuracy": acc,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "cpu_ram_mb": self.cpu_ram_mb
                },
                f,
                indent=4
            )

        return acc, precision, recall, f1

    # ==================================================
    # CONFUSION MATRIX
    # ==================================================

    def _save_confusion_matrix(
            self,
            y_true,
            y_pred,
            class_names=None
    ):

        cm = confusion_matrix(y_true, y_pred)

        fig, ax = plt.subplots(figsize=(10, 10))

        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=class_names
        )

        disp.plot(
            ax=ax,
            cmap="Blues",
            xticks_rotation=45,
            colorbar=False
        )

        plt.title(
            f"Confusion Matrix — {self.classifier_name.upper()}"
        )

        plt.tight_layout()

        plt.savefig(
            os.path.join(self.save_dir, "confusion_matrix.png")
        )

        plt.close()


# ==================================================
# TRAIN FUNCTIONS
# ==================================================

def train_svm(dataset_path, num_classes=10, batch_size=32, save_dir="results/svm"):
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)

    train_loader, val_loader, test_loader, classes = get_dataloaders(
        dataset_path=dataset_path,
        train_transform=train_transform,
        test_transform=test_transform,
        batch_size=batch_size
    )

    trainer = MLTrainer(
        classifier_name="svm",
        device=device,
        save_dir=save_dir,
        kernel="rbf",
        C=1.0
    )

    trainer.fit(train_loader, val_loader)
    trainer.evaluate_test(test_loader, class_names=classes)


def train_knn(dataset_path, num_classes=10, batch_size=32, save_dir="results/knn"):
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)

    train_loader, val_loader, test_loader, classes = get_dataloaders(
        dataset_path=dataset_path,
        train_transform=train_transform,
        test_transform=test_transform,
        batch_size=batch_size
    )

    trainer = MLTrainer(
        classifier_name="knn",
        device=device,
        save_dir=save_dir,
        n_neighbors=5
    )

    trainer.fit(train_loader, val_loader)
    trainer.evaluate_test(test_loader, class_names=classes)


def train_random_forest(dataset_path, num_classes=10, batch_size=32, save_dir="results/random_forest"):
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)

    train_loader, val_loader, test_loader, classes = get_dataloaders(
        dataset_path=dataset_path,
        train_transform=train_transform,
        test_transform=test_transform,
        batch_size=batch_size
    )

    trainer = MLTrainer(
        classifier_name="random_forest",
        device=device,
        save_dir=save_dir,
        n_estimators=100
    )

    trainer.fit(train_loader, val_loader)
    trainer.evaluate_test(test_loader, class_names=classes)


# ==================================================
# MAIN
# ==================================================

if __name__ == "__main__":
    import os

    BASE_DIR = os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )

    DATASET_PATH = os.path.join(BASE_DIR, "dataset", "raw-img")

    train_svm(
        dataset_path=DATASET_PATH,
        num_classes=10,
        batch_size=32,
        save_dir=os.path.join(BASE_DIR, "results", "svm")
    )

    train_knn(
        dataset_path=DATASET_PATH,
        num_classes=10,
        batch_size=32,
        save_dir=os.path.join(BASE_DIR, "results", "knn")
    )

    train_random_forest(
        dataset_path=DATASET_PATH,
        num_classes=10,
        batch_size=32,
        save_dir=os.path.join(BASE_DIR, "results", "random_forest")
    )