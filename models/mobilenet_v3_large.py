import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import models

from training.trainer import Trainer
from data.dataset_loader import get_dataloaders, train_transform, test_transform


# ==================================================
# MODEL
# ==================================================

class MobileNetV3LargeClassifier(nn.Module):

    def __init__(
            self,
            num_classes=37,
            pretrained=True,
            dropout_rate=0.3
    ):

        super().__init__()

        if pretrained:

            self.backbone = models.mobilenet_v3_large(
                weights=models.MobileNet_V3_Large_Weights.DEFAULT
            )

        else:

            self.backbone = models.mobilenet_v3_large(
                weights=None
            )

        # MobileNetV3 Large: classifier gồm
        # Linear(960, 1280) → Hardswish → Dropout → Linear(1280, 1000)
        # Thay Linear cuối bằng head của mình
        in_features = self.backbone.classifier[-1].in_features

        self.backbone.classifier[-1] = nn.Sequential(

            nn.Dropout(
                dropout_rate
            ),

            nn.Linear(
                in_features,
                num_classes
            )
        )

    def forward(self, x):

        return self.backbone(x)

    def freeze_backbone(self):
        """Đóng băng toàn bộ features, chỉ train classifier."""

        for param in self.backbone.parameters():
            param.requires_grad = False

        for param in self.backbone.classifier.parameters():
            param.requires_grad = True

    def unfreeze_from(self, layer_index=100):
        """Mở các parameters từ layer_index trở đi."""

        all_params = list(self.backbone.parameters())

        for param in all_params[:layer_index]:
            param.requires_grad = False

        for param in all_params[layer_index:]:
            param.requires_grad = True

        # Classifier luôn được train
        for param in self.backbone.classifier.parameters():
            param.requires_grad = True

# ==================================================
# TRAIN FUNCTION
# ==================================================

def train_mobilenet_v3_large(
        dataset_path,
        num_classes=37,
        batch_size=32,
        epochs=20,
        fine_tune_epoch=5,
        patience=5,
        save_dir="results/mobilenet_v3_large"
):

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("Device:", device)

    # --------------------------------------------------
    # Dataset
    # --------------------------------------------------

    train_loader, val_loader, test_loader, classes = get_dataloaders(
        dataset_path=dataset_path,
        train_transform=train_transform,
        test_transform=test_transform,
        batch_size=batch_size
    )

    print(f"Classes ({len(classes)}): {classes}")

    # --------------------------------------------------
    # Model
    # --------------------------------------------------

    model = MobileNetV3LargeClassifier(
        num_classes=num_classes,
        pretrained=True,
        dropout_rate=0.3
    ).to(device)

    # Giai đoạn 1: frozen backbone, chỉ train classifier
    model.freeze_backbone()

    # --------------------------------------------------
    # Optimizer / Scheduler / Criterion
    # --------------------------------------------------

    optimizer = optim.AdamW(

        filter(
            lambda p: p.requires_grad,
            model.parameters()
        ),

        lr=1e-3,
        weight_decay=1e-4
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs
    )

    criterion = nn.CrossEntropyLoss()

    # --------------------------------------------------
    # Trainer
    # --------------------------------------------------

    trainer = Trainer(

        model=model,

        train_loader=train_loader,

        val_loader=val_loader,

        criterion=criterion,

        optimizer=optimizer,

        scheduler=scheduler,

        device=device,

        save_dir=save_dir,

        patience=patience
    )

    # --------------------------------------------------
    # Fine-tune callback
    # --------------------------------------------------

    def start_finetune():

        print("\nUnfreezing Backbone...\n")

        model.unfreeze_from(layer_index=100)

        # Giảm lr xuống 10x khi fine-tune toàn bộ
        trainer.optimizer = optim.AdamW(
            model.parameters(),
            lr=1e-4,
            weight_decay=1e-4
        )

        # Reset scheduler theo số epoch còn lại
        trainer.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            trainer.optimizer,
            T_max=epochs - fine_tune_epoch
        )

    # --------------------------------------------------
    # Train
    # --------------------------------------------------

    trainer.fit(
        epochs=epochs,
        fine_tune_epoch=fine_tune_epoch,
        fine_tune_callback=start_finetune
    )

    trainer.evaluate_test(
        test_loader,
        class_names=classes
    )


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

    train_mobilenet_v3_large(
        dataset_path=os.path.join(BASE_DIR, "dataset", "raw-img"),
        num_classes=10,
        batch_size=32,
        epochs=20,
        fine_tune_epoch=5,
        patience=5,
        save_dir=os.path.join(BASE_DIR, "results", "mobilenet_v3_large")
    )
