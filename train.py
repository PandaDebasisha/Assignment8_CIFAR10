import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from torchsummary import summary
from torch.optim.lr_scheduler import OneCycleLR

from models.custom_model import CIFAR10Net,CIFAR10Net2, CIFAR10Net3
from utils.transforms import get_train_transforms, get_test_transforms

class AlbumentationDataset:
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # Convert PIL Image to numpy array
        image = np.array(self.data[idx][0])
        label = self.data[idx][1]
        
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed["image"]
            
        return image, label

def train(model, device, train_loader, optimizer, criterion, scheduler, epoch):
    model.train()
    pbar = tqdm(train_loader)
    train_loss = 0
    correct = 0
    processed = 0

    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        train_loss += loss.item()
        
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        processed += len(data)
        
        current_lr = scheduler.get_last_lr()[0]
        pbar.set_description(
            desc=f'Epoch={epoch} Loss={loss.item():.4f} Batch_id={batch_idx} '
            f'Accuracy={100*correct/processed:0.2f} LR={current_lr:.6f}'
        )

def test(model, device, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)\n')
    return accuracy

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load CIFAR10 dataset
    trainset = datasets.CIFAR10(root='./data', train=True, download=True)
    testset = datasets.CIFAR10(root='./data', train=False, download=True)
    
    # Apply transforms
    train_dataset = AlbumentationDataset(trainset, get_train_transforms())
    test_dataset = AlbumentationDataset(testset, get_test_transforms())
    
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=2)
    
    model = CIFAR10Net3().to(device)
    
    # Print model summary
    summary(model, input_size=(3, 32, 32))
    
    # Calculate total parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f'\nTotal Parameters: {total_params}')
    
    criterion = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
    
    # Calculate steps per epoch
    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * 200  # 200 epochs
    
    # Add OneCycleLR scheduler
    scheduler = OneCycleLR(
        optimizer,
        max_lr=0.1,  # maximum learning rate
        epochs=200,
        steps_per_epoch=steps_per_epoch,
        pct_start=0.3,  # spend 30% of the time warming up
        anneal_strategy='cos',  # cosine annealing
        div_factor=25.0,  # initial_lr = max_lr/div_factor
        final_div_factor=1e4,  # min_lr = initial_lr/final_div_factor
    )
    
    # Training loop
    epochs = 200
    best_acc = 0
    for epoch in range(1, epochs + 1):
        train(model, device, train_loader, optimizer, criterion, scheduler, epoch)
        test_acc = test(model, device, test_loader, criterion)
        
        # Save best model
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), 'best_model.pth')
            print(f'Best accuracy: {best_acc:.2f}%')

if __name__ == '__main__':
    main() 