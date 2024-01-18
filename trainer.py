from torchmetrics import Accuracy
from torch import nn
from torch.optim import Adam
from tqdm import tqdm


class Trainer:
    def __init__(self, model):
        self.model = model
        self.metric = Accuracy(task="multiclass", num_classes=model.num_classes)
        self.criterion = nn.CrossEntropyLoss()

    def configure_optimizers(self):
        return Adam(self.model.parameters(), lr=0.0001)

    def training_step(self, batch, device):
        self.model.train()
        images, labels = batch
        images = images.to(device)
        labels = labels.to(device)

        output = self.model(images)
        loss = self.criterion(output, labels)
        metrics = self.metric(output, labels)

        self.configure_optimizers().zero_grad()
        loss.backward()
        self.configure_optimizers().step()

        return loss, metrics

    def validation_step(self, batch, device):
        self.model.eval()
        images, labels = batch
        images = images.to(device)
        labels = labels.to(device)

        output = self.model(images)
        loss = self.criterion(output, labels)
        metrics = self.metric(output, labels)

        return loss, metrics

    def fit(self, max_epoch, train_loader, val_loader, device):
        loss = 0
        metrics = 0
        val_loss = 0
        val_metrics = 0
        self.model.to(device)

        for epoch in range(max_epoch):
            print(f"___ Epoch {epoch + 1}/{max_epoch} ___")
            for batch in tqdm(train_loader):
                loss, metrics = self.training_step(batch, device)
            print(f"Training: Loss: {loss:.4f}, Accuracy: {metrics:.4f}")

            for val_batch in tqdm(val_loader):
                val_loss, val_metrics = self.validation_step(val_batch, device)
            print(f"Validation: Loss: {val_loss:.4f}, Accuracy: {val_metrics:.4f}\n")
