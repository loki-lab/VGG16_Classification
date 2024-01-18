from torch import nn
from torch.optim import Adam
from tqdm import tqdm
import torch


class Trainer:
    def __init__(self, model, device="cpu"):
        self.model = model
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        self.best_metric = 0

    def configure_optimizers(self):
        return Adam(self.model.parameters(), lr=0.0001)

    @staticmethod
    def get_total_result(output, target, loss):
        total_loss = 0
        total_correct = 0
        _, predicted = torch.max(output, 1)
        total_correct += predicted.eq(target).sum().item()
        total_loss += loss.item()
        return total_loss, total_correct

    def training_step(self, batch):
        self.model.train()
        images, labels = batch
        images = images.to(self.device)
        labels = labels.to(self.device)

        output = self.model(images)
        loss = self.criterion(output, labels)

        self.configure_optimizers().zero_grad()
        loss.backward()
        self.configure_optimizers().step()
        total_loss, total_correct = self.get_total_result(output, labels, loss)

        return total_loss, total_correct

    def validation_step(self, batch):
        self.model.eval()
        images, labels = batch
        images = images.to(self.device)
        labels = labels.to(self.device)

        output = self.model(images)
        loss = self.criterion(output, labels)
        total_loss, total_correct = self.get_total_result(output, labels, loss)

        return total_loss, total_correct

    def training(self, train_loader):
        total_correct = 0
        total_loss = 0

        for batch in tqdm(train_loader):
            total_loss, total_correct = self.training_step(batch)
        accuracy = total_correct / len(train_loader)
        loss_in_epoch = total_loss / len(train_loader)

        return accuracy, loss_in_epoch

    def validation(self, val_loader):
        val_total_correct = 0
        val_total_loss = 0

        for batch in tqdm(val_loader):
            val_total_loss, val_total_correct = self.training_step(batch)
        accuracy = val_total_correct / len(val_loader)
        loss_in_epoch = val_total_loss / len(val_loader)

        return accuracy, loss_in_epoch

    def save_checkpoint(self, loss, accuracy, epoch, model, optimizer):
        if accuracy > self.best_metric:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
            }, "./checkpoints/best_checkpoint.pth")
            self.best_metric = accuracy
            print("Saved best checkpoint")

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        }, "./checkpoints/last_checkpoint.pth")

    def fit(self, max_epoch, train_loader, val_loader):

        self.model.to(self.device)
        for epoch in range(max_epoch):
            print(f"___ Epoch {epoch + 1}/{max_epoch} ___")

            print("Training:")
            loss, metrics = self.training(train_loader)
            print(f"Training result: Loss: {loss:.4f}, Accuracy: {metrics:.4f}")

            print("Validating:")
            val_loss, val_metrics = self.validation(val_loader)
            print(f"Validation result: Loss: {val_loss:.4f}, Accuracy: {val_metrics:.4f}\n")

            self.save_checkpoint(epoch, val_loss, val_metrics, self.model, self.configure_optimizers())
