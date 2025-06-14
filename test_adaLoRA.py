import warnings
warnings.simplefilter("ignore")
from typing import Sequence, Type, Union

import torch
import torchvision
import torchvision.transforms as transforms

from torchvision.datasets import CIFAR10

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# from src.loralib.layers import LoRALayer
# from src.loralib.adalora import SVDLinear
from src.loralib import (RankAllocator,
                         compute_orth_regu,
                         mark_only_lora_as_trainable,
                         LoRALayer,
                         SVDLinear,
                         SVDConv2d)

from tqdm.notebook import tqdm

from opacus.validators import ModuleValidator

from opacus import PrivacyEngine

# from opacus import PrivacyEngine
from src.privacy_engine import PrivacyEngineAugmented

import numpy as np
from opacus.utils.batch_memory_manager import BatchMemoryManager


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        # self.conv2 = nn.Conv2d(6, 16, 5)
        self.conv2 = SVDConv2d(6, 16, 5, r = 16)
        # self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc1 = SVDLinear(16 * 5 * 5, 120, r=12)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def accuracy(preds, labels):
    return (preds == labels).mean()

def train(model, train_loader, optimizer, epoch, device,MAX_PHYSICAL_BATCH_SIZE, DELTA, privacy_engine):
    model.train()
    criterion = nn.CrossEntropyLoss()

    losses = []
    top1_acc = []

    # Initialize the RankAllocator
    rankallocator = RankAllocator(
        model, lora_r=12, target_rank=8,  #12，8
        init_warmup=500, final_warmup=1500, mask_interval=10,
        total_step=3000, beta1=0.85, beta2=0.85,
    )

    with BatchMemoryManager(
        data_loader=train_loader,
        max_physical_batch_size=MAX_PHYSICAL_BATCH_SIZE,
        optimizer=optimizer
    ) as memory_safe_data_loader:

        for i, (images, target) in enumerate(memory_safe_data_loader):
            optimizer.zero_grad()
            images = images.to(device)
            target = target.to(device)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            preds = np.argmax(output.detach().cpu().numpy(), axis=1)
            labels = target.detach().cpu().numpy()

            # measure accuracy and record loss
            acc = accuracy(preds, labels)

            losses.append(loss.item())
            top1_acc.append(acc)

            # loss.backward()
            (loss+compute_orth_regu(model, regu_weight=0.1)).backward()
            optimizer.step()
            # add
            rankallocator.update_and_mask(model, epoch)

            if (i+1) % 200 == 0:
                epsilon = privacy_engine.get_epsilon(DELTA)
                print(
                    f"\t******************************************Train Epoch: {epoch} \t"
                    f"Loss: {np.mean(losses):.6f} "
                    f"Acc@1: {np.mean(top1_acc) * 100:.6f} "
                    f"(ε = {epsilon:.2f}, δ = {DELTA})"
                )

def test(model, test_loader, device):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    losses = []
    top1_acc = []

    with torch.no_grad():
        for images, target in test_loader:
            images = images.to(device)
            target = target.to(device)

            output = model(images)
            loss = criterion(output, target)
            preds = np.argmax(output.detach().cpu().numpy(), axis=1)
            labels = target.detach().cpu().numpy()
            acc = accuracy(preds, labels)

            losses.append(loss.item())
            top1_acc.append(acc)

    top1_avg = np.mean(top1_acc)

    print(
        f"\tTest set:"
        f"Loss: {np.mean(losses):.6f} "
        f"Acc: {top1_avg * 100:.6f} "
    )
    return np.mean(top1_acc)

def main():
    MAX_GRAD_NORM = 1.2
    EPSILON = 50.0
    DELTA = 1e-5
    EPOCHS = 20

    LR = 1e-3

    BATCH_SIZE = 512
    MAX_PHYSICAL_BATCH_SIZE = 128

    CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
    CIFAR10_STD_DEV = (0.2023, 0.1994, 0.2010)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD_DEV),
    ])

    DATA_ROOT = './cifar10'

    train_dataset = CIFAR10(
        root=DATA_ROOT, train=True, download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
    )

    test_dataset = CIFAR10(
        root=DATA_ROOT, train=False, download=True, transform=transform)

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    model = Net()
    errors = ModuleValidator.validate(model, strict=False)
    errors[-5:]

    model = ModuleValidator.fix(model)
    ModuleValidator.validate(model, strict=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)

    # mark_only_lora_as_trainable(model)

    for name, param in model.named_parameters():
        print(name, param.requires_grad)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.RMSprop(model.parameters(), lr=LR)

    # privacy_engine = PrivacyEngine()
    privacy_engine = PrivacyEngineAugmented()

    model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
        module=model,
        optimizer=optimizer,
        data_loader=train_loader,
        epochs=EPOCHS,
        target_epsilon=EPSILON,
        target_delta=DELTA,
        max_grad_norm=MAX_GRAD_NORM,
    )
    print(f"Using sigma={optimizer.noise_multiplier} and C={MAX_GRAD_NORM}")

    for epoch in range(EPOCHS): #tqdm(range(EPOCHS), desc="Epoch", unit="epoch")
        train(model, train_loader, optimizer, epoch + 1, device, MAX_PHYSICAL_BATCH_SIZE, DELTA, privacy_engine)

if __name__ == "__main__":
    main()
    # def register_grad_sampler(
    #     target_class_or_classes: Union[Type[nn.Module], Sequence[Type[nn.Module]]]
    # ):
    #     print("OK")
    # if isinstance(SVDLinear(16 * 5 * 5, 120, r=12), nn.Module): #,Sequence
    #     print("Yes!")
    # else:
    #     print("No?")