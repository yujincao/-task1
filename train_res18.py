import timm
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from data import BirdDataset, train_transform, test_transform
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Load the model
model = timm.create_model('resnet18', pretrained=True, num_classes=200)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Load the dataset
train_data = BirdDataset(root_dir="dataset/CUB_200_2011/images", mode="train", transform=train_transform)
print(len(train_data))
test_data = BirdDataset(root_dir="dataset/CUB_200_2011/images", mode="test", transform=test_transform)
print(len(test_data))
train_loader = DataLoader(train_data, batch_size=128, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)


# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
#scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

#tensorboard
store_path = 'runs/res18'
if not os.path.exists(store_path):
    os.makedirs(store_path)
writer = SummaryWriter(store_path)

# Train the model并且记录训练过程的损失和准确率
epochs = 15
total_step = len(train_loader)
for epoch in range(epochs):
    model.train()
    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #scheduler.step()
        if (i+1) % 10 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch+1, epoch, i+1, total_step, loss.item()))
            writer.add_scalar('train_loss', loss.item(), epoch * len(train_loader) + i)
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            if (i+1) % 10 == 0:
                writer.add_scalar('test_loss', loss.item(), epoch * len(test_loader) + i)
        print('Test Accuracy of the model on the test images: {} %'.format(100 * correct / total))
        writer.add_scalar('test_accuracy', 100 * correct / total, epoch)

    
    # Save the model checkpoint
    torch.save(model.state_dict(), os.path.join(store_path, f'model_{epoch}_{round(100 * correct / total, 2)}.ckpt'))

writer.close()





