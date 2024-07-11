import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
from torchvision.io import read_video
from torchvision.models.video import r2plus1d_18
import torchvision.transforms as transforms
from sklearn.metrics import classification_report, f1_score
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('test/uni-class_no_duplicates.csv')

df.replace({'yes': 1, 'no': 0}, inplace=True)
df.to_csv('test/uni-class_no_duplicates_numeric.csv', index=False)

class VideoDataset(Dataset):
    def __init__(self, csv_file, video_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.video_dir = video_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        video_name = self.data.iloc[idx, 0]
        video_path = os.path.join(self.video_dir, video_name + '.mp4')
        
        try:
            video, _, _ = read_video(video_path, pts_unit='sec')
        except FileNotFoundError:
            new_idx = torch.randint(0, len(self.data), (1,)).item()
            return self.__getitem__(new_idx)
        
        video = video[:16]
        video = video.permute(3, 0, 1, 2)
        
        if self.transform:
            video = self.transform(video)
        
        labels = torch.tensor(self.data.iloc[idx, 1:].values.astype(float)).float()

        return video, labels

class NormalizeVideo(transforms.Normalize):
    def forward(self, tensor):
        for t in range(tensor.shape[1]):
            tensor[:, t] = super().forward(tensor[:, t])
        return tensor

transform = transforms.Compose([
    transforms.Lambda(lambda x: x / 255.0), 
    transforms.Resize((112, 112), antialias=True),
    NormalizeVideo(mean=[0.43216, 0.394666, 0.37645], std=[0.22803, 0.22145, 0.216989])
])

dataset = VideoDataset('test/uni-class_no_duplicates_numeric.csv', 'destination_videos-20240710T175151Z-001/destination_videos', transform=transform)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

class VideoClassifier(nn.Module):
    def __init__(self, num_classes):
        super(VideoClassifier, self).__init__()
        self.backbone = r2plus1d_18(pretrained=True)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        if x.dim() == 4:
            x = x.unsqueeze(2)
        x = self.backbone(x)
        return self.sigmoid(x)

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE_loss = nn.BCELoss(reduction='none')(inputs, targets)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
        return torch.mean(F_loss)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VideoClassifier(num_classes=7).to(device)
criterion = FocalLoss()
optimizer = optim.Adam(model.parameters(), lr=0.00001)

num_epochs = 10
losses = []
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0.0
    all_labels = []
    all_preds = []
    for videos, labels in dataloader:
        videos, labels = videos.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(videos)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        
        predicted = (outputs > 0.5).float()
        all_labels.append(labels.cpu().numpy())
        all_preds.append(predicted.cpu().numpy())
    
    epoch_loss /= len(dataloader)
    losses.append(epoch_loss)
    
    all_labels = np.concatenate(all_labels)
    all_preds = np.concatenate(all_preds)
    
    accuracy = (all_preds == all_labels).mean()
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=1)
    
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.4f}, F1-score: {f1:.4f}')

plt.figure()
plt.plot(range(1, num_epochs+1), losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss over Epochs')
plt.savefig('test/graphs/training_loss_2d1.png')

model.eval()
all_labels = []
all_preds = []
with torch.no_grad():
    for videos, labels in dataloader:
        videos, labels = videos.to(device), labels.to(device)
        outputs = model(videos)
        predicted = (outputs > 0.5).float()
        all_labels.append(labels.cpu().numpy())
        all_preds.append(predicted.cpu().numpy())

all_labels = np.concatenate(all_labels)
all_preds = np.concatenate(all_preds)

report = classification_report(all_labels, all_preds, target_names=[f'Class {i}' for i in range(7)], zero_division=1)
with open('test/reports/classification_report_2d1.txt', 'w') as f:
    f.write(report)
