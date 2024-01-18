from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import random_split, DataLoader
from trainer import Trainer
from model import VGG16
import torch


if __name__ == "__main__":

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    path = "./PetImg"
    train_size = 20000
    val_size = 5000

    transforms = transforms.Compose([transforms.ToTensor(),
                                     transforms.Resize((224, 224), antialias=True),
                                     transforms.Normalize((0.5,), (0.5,)),
                                     transforms.RandomPerspective(distortion_scale=0.5, p=0.1),
                                     transforms.RandomRotation(degrees=(0, 180)),
                                     transforms.RandomHorizontalFlip(p=0.5),
                                     transforms.RandomVerticalFlip(p=0.5)])

    dataset = ImageFolder(path, transform=transforms)
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_dl = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=2)
    val_dl = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=2)

    model = VGG16(num_classes=2)

    trainer = Trainer(model, device=device)
    trainer.fit(max_epoch=30, train_loader=train_dl, val_loader=val_dl)
