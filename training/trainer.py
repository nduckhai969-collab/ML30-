import os
import json
import time
import psutil
import torch
import matplotlib.pyplot as plt
from torch.amp import autocast, GradScaler
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay)
class Trainer:
    def __init__(
            self,
            model,
            train_loader,
            val_loader,
            criterion,
            optimizer,
            scheduler,
            device,
            save_dir,
            patience=5):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.save_dir = save_dir
        self.patience = patience
        os.makedirs(
            self.save_dir,
            exist_ok=True)
        self.scaler = GradScaler(
            device="cuda",
            enabled=torch.cuda.is_available())
        self.best_acc = 0.0
        self.counter = 0
        self.best_loss = float("inf")
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
        self.training_time = 0
        self.inference_time = 0
        self.end_to_end_time = 0
        self.model_params = 0
        self.model_size_mb = 0
        self.cpu_ram_mb = 0
        self.gpu_memory_mb = 0
    def train_one_epoch(self):
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        loader = tqdm(
            self.train_loader,
            leave=False)
        for images, labels in loader:
            images = images.to(self.device)
            labels = labels.to(self.device)
            self.optimizer.zero_grad()
            with autocast(
                device_type="cuda",
                enabled=torch.cuda.is_available()
            ):
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            running_loss += loss.item()
            preds = outputs.argmax(dim=1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
            loader.set_postfix(loss=f"{loss.item():.4f}")
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100.0 * correct / total
        return epoch_loss, epoch_acc
    def validate(self):
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in self.val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                running_loss += loss.item()
                preds = outputs.argmax(dim=1)
                total += labels.size(0)
                correct += (preds == labels).sum().item()
        val_loss = running_loss / len(self.val_loader)
        val_acc = 100.0 * correct / total
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            torch.save(
                self.model.state_dict(),
                os.path.join(self.save_dir,"best_loss_model.pth"))
        return val_loss, val_acc
    def evaluate_test(
            self,
            test_loader,
            class_names=None):
        self.model.eval()
        preds_list = []
        labels_list = []
        with torch.no_grad():
            for images, labels in tqdm(
                    test_loader,
                    desc="Testing"):
                images = images.to(self.device)
                outputs = self.model(images)
                preds = outputs.argmax(dim=1)
                preds_list.extend(preds.cpu().numpy())
                labels_list.extend(labels.numpy())
        acc = accuracy_score(labels_list, preds_list)
        precision = precision_score(
            labels_list,
            preds_list,
            average="weighted")
        recall = recall_score(
            labels_list,
            preds_list,
            average="weighted")
        f1 = f1_score(
            labels_list,
            preds_list,
            average="weighted")
        self.save_confusion_matrix(
            labels_list,
            preds_list,
            class_names)
        self.measure_inference_time(test_loader)
        print("\nTEST")
        print(f"Accuracy         : {acc*100:.2f}%")
        print(f"Precision        : {precision:.4f}")
        print(f"Recall           : {recall:.4f}")
        print(f"F1-score         : {f1:.4f}")
        print(f"Inference   (ms) : {self.inference_time:.4f}")
        print(f"End-to-end  (ms) : {self.end_to_end_time:.4f}")
        with open(
            os.path.join(self.save_dir, "test_metrics.json"),
            "w"
        ) as f:
            json.dump(
                {
                    "accuracy": acc,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "inference_ms": self.inference_time,
                    "end_to_end_ms": self.end_to_end_time
                },
                f,
                indent=4)
        return acc, precision, recall, f1
    def calculate_model_parameters(self):
        self.model_params = sum(
            p.numel()
            for p in self.model.parameters())
        return self.model_params
    def calculate_model_size(self):
        model_path = os.path.join(
            self.save_dir,
            "best_acc_model.pth")
        if os.path.exists(model_path):
            self.model_size_mb = (
                os.path.getsize(model_path) / 1024 / 1024
            )
        return self.model_size_mb
    def measure_cpu_ram(self):
        process = psutil.Process()
        self.cpu_ram_mb = (
            process.memory_info().rss / 1024 / 1024
        )
        return self.cpu_ram_mb
    def measure_gpu_memory(self):
        if torch.cuda.is_available():
            self.gpu_memory_mb = (
                torch.cuda.max_memory_allocated() / 1024 / 1024)
        else:
            self.gpu_memory_mb = 0
        return self.gpu_memory_mb
    def measure_inference_time(
            self,
            loader,
            num_batches=30):
        self.model.eval()
        total_images = 0
        total_inference_ms = 0.0
        total_e2e_ms = 0.0
        with torch.no_grad():
            for idx, (images, _) in enumerate(loader):
                if idx >= num_batches:
                    break
                batch_size = images.size(0)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                e2e_start = time.perf_counter()
                images = images.to(self.device)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                infer_start = time.perf_counter()
                _ = self.model(images)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                infer_end = time.perf_counter()
                e2e_end = infer_end
                total_inference_ms += (
                    (infer_end - infer_start) * 1000)
                total_e2e_ms += (
                    (e2e_end - e2e_start) * 1000)
                total_images += batch_size
        self.inference_time = total_inference_ms / total_images
        self.end_to_end_time = total_e2e_ms / total_images
        return self.inference_time, self.end_to_end_time
    def save_curves(self):
        plt.figure()
        plt.plot(self.train_accs, label="Train Accuracy")
        plt.plot(self.val_accs, label="Validation Accuracy")
        plt.legend()
        plt.grid(True)
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy (%)")
        plt.title("Accuracy Curve")
        plt.savefig(os.path.join(self.save_dir, "accuracy_curve.png"))
        plt.close()
        plt.figure()
        plt.plot(self.train_losses, label="Train Loss")
        plt.plot(self.val_losses, label="Validation Loss")
        plt.legend()
        plt.grid(True)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Loss Curve")
        plt.savefig(os.path.join(self.save_dir, "loss_curve.png"))
        plt.close()
    def save_confusion_matrix(
            self,
            y_true,
            y_pred,
            class_names=None):
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots(figsize=(10, 10))
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=class_names)
        disp.plot(
            ax=ax,
            cmap="Blues",
            xticks_rotation=45,
            colorbar=False)
        plt.title("Confusion Matrix")
        plt.tight_layout()
        plt.savefig(
            os.path.join(self.save_dir, "confusion_matrix.png"))
        plt.close()
    def save_metrics(self):
        metrics = {
            "best_val_accuracy": self.best_acc,
            "best_val_loss": self.best_loss,
            "training_time_seconds": self.training_time,
            "parameters": self.model_params,
            "model_size_mb": self.model_size_mb,
            "cpu_ram_mb": self.cpu_ram_mb,
            "gpu_memory_mb": self.gpu_memory_mb,
            "train_acc": self.train_accs,
            "val_acc": self.val_accs,
            "train_loss": self.train_losses,
            "val_loss": self.val_losses
        }
        with open(
            os.path.join(self.save_dir, "metrics.json"),
            "w"
        ) as f:
            json.dump(metrics, f, indent=4)
    def fit(
            self,
            epochs,
            fine_tune_epoch=None,
            fine_tune_callback=None):
        print("\nTraining Started...\n")
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        start_time = time.perf_counter()
        for epoch in range(epochs):
            if (
                fine_tune_epoch is not None
                and epoch == fine_tune_epoch
                and fine_tune_callback is not None):
                fine_tune_callback()
                print("\nFine-tuning Started\n")
            train_loss, train_acc = self.train_one_epoch()
            val_loss, val_acc = self.validate()
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accs.append(train_acc)
            self.val_accs.append(val_acc)
            lr = self.optimizer.param_groups[0]["lr"]
            print(
                f"Epoch [{epoch+1}/{epochs}] | "
                f"LR={lr:.6f} | "
                f"Train Loss={train_loss:.4f} | "
                f"Train Acc={train_acc:.2f}% | "
                f"Val Loss={val_loss:.4f} | "
                f"Val Acc={val_acc:.2f}%")
            if self.scheduler is not None:
                self.scheduler.step()
            if val_acc > self.best_acc:
                self.best_acc = val_acc
                self.counter = 0
                torch.save(
                    self.model.state_dict(),
                    os.path.join(
                        self.save_dir,
                        "best_acc_model.pth"))
            else:
                self.counter += 1
            if self.counter >= self.patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break
        self.training_time = (time.perf_counter() - start_time)
        self.calculate_model_parameters()
        self.calculate_model_size()
        self.measure_cpu_ram()
        self.measure_gpu_memory()
        self.save_curves()
        self.save_metrics()
        best_model_path = os.path.join(
            self.save_dir,
            "best_acc_model.pth")
        self.model.load_state_dict(
            torch.load(
                best_model_path,
                map_location=self.device,
                weights_only=True ))
        print("\nTraining Finished")
        print(f"Best Val Accuracy : {self.best_acc:.2f}%")
        print(f"Training Time     : {self.training_time:.2f}s")
        print(f"Parameters        : {self.model_params:,}")
        print(f"Model Size        : {self.model_size_mb:.2f} MB")
        print(f"CPU RAM           : {self.cpu_ram_mb:.2f} MB")
        print(f"GPU Memory        : {self.gpu_memory_mb:.2f} MB")