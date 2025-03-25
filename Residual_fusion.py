import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import mean_absolute_error, mean_squared_error

class MultiModalDataset(Dataset):
    def __init__(self, text_dir, audio_dir, video_dir, label_file):
        self.text_files = sorted([os.path.join(text_dir, f) for f in os.listdir(text_dir) if f.endswith('.npy')])
        self.audio_files = sorted([os.path.join(audio_dir, f) for f in os.listdir(audio_dir) if f.endswith('.npy')])
        self.video_files = sorted([os.path.join(video_dir, f) for f in os.listdir(video_dir) if f.endswith('.npy')])

        self.labels = pd.read_csv(label_file)['PHQ8_Score'].values.astype(np.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        text = torch.tensor(np.load(self.text_files[idx]), dtype=torch.float32)
        audio = torch.tensor(np.load(self.audio_files[idx]), dtype=torch.float32)
        video = torch.tensor(np.load(self.video_files[idx]), dtype=torch.float32)

        label = torch.tensor(self.labels[idx], dtype=torch.float32).unsqueeze(0)

        return text, audio, video, label


class MultiModalModel(nn.Module):
    def __init__(self):
        super(MultiModalModel, self).__init__()

        # Text modal residual block
        self.text_conv1 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.text_bn1 = nn.BatchNorm2d(128)
        self.text_conv2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.text_bn2 = nn.BatchNorm2d(128)
        self.text_proj = nn.Conv2d(256, 128, kernel_size=1)  

        # Audio modal residual block
        self.audio_conv1 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.audio_bn1 = nn.BatchNorm2d(128)
        self.audio_conv2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.audio_bn2 = nn.BatchNorm2d(128)
        self.audio_proj = nn.Conv2d(256, 128, kernel_size=1) 

        # Video modal residual block
        self.video_conv1 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.video_bn1 = nn.BatchNorm2d(128)
        self.video_conv2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.video_bn2 = nn.BatchNorm2d(128)
        self.video_proj = nn.Conv2d(256, 128, kernel_size=1)  

        # Weight parameters (learnable)
        self.alpha = nn.Parameter(torch.tensor(0.33))
        self.beta = nn.Parameter(torch.tensor(0.33))
        self.gamma = nn.Parameter(torch.tensor(0.34))

        self.fc1 = nn.Linear(128 * 28 * 28, 512)
        self.fc2 = nn.Linear(512, 1)

    def forward(self, text, audio, video):
        # Text residual
        res_text = F.relu(self.text_bn1(self.text_conv1(text)))
        res_text = self.text_bn2(self.text_conv2(res_text)) + self.text_proj(text)  
        res_text = F.relu(res_text)

        # Audio residual
        res_audio = F.relu(self.audio_bn1(self.audio_conv1(audio)))
        res_audio = self.audio_bn2(self.audio_conv2(res_audio)) + self.audio_proj(audio)  
        res_audio = F.relu(res_audio)

        # Video residual
        res_video = F.relu(self.video_bn1(self.video_conv1(video)))
        res_video = self.video_bn2(self.video_conv2(res_video)) + self.video_proj(video)  
        res_video = F.relu(res_video)

        # Modal fusion
        fused = self.alpha * res_text + self.beta * res_audio + self.gamma * res_video

        fused = fused.view(fused.size(0), -1)

        out = F.relu(self.fc1(fused))
        out = self.fc2(out)
        return out

def main():
    text_dir = r'D:\DAIC-WOZ\parameter\text'
    audio_dir = r'D:\DAIC-WOZ\parameter\audio'
    video_dir = r'D:\DAIC-WOZ\parameter\visual'
    label_file = r'D:\DAIC-WOZ\train_split_Depression_AVEC2017.csv'

    batch_size = 16
    dataset = MultiModalDataset(text_dir, audio_dir, video_dir, label_file)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MultiModalModel().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 100

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        all_preds = []
        all_labels = []

        for i, (text, audio, video, labels) in enumerate(dataloader):
            text, audio, video, labels = text.to(device), audio.to(device), video.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model(text, audio, video)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            all_preds.extend(outputs.detach().cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        all_preds = np.array(all_preds).flatten()
        all_labels = np.array(all_labels).flatten()
        mae = mean_absolute_error(all_labels, all_preds)
        rmse = np.sqrt(mean_squared_error(all_labels, all_preds))

        print(f'Epoch [{epoch+1}/{num_epochs}], MAE: {mae:.4f}, RMSE: {rmse:.4f}')

    # Inference section, printing real labels and predicted labels
    model.eval()
    all_preds = []
    all_labels = []
    total_mae = 0.0
    total_rmse = 0.0
    sample_count = 0

    with torch.no_grad():
        for i, (text, audio, video, labels) in enumerate(dataloader):
            text, audio, video, labels = text.to(device), audio.to(device), video.to(device), labels.to(device)
            outputs = model(text, audio, video)

            preds = outputs.detach().cpu().numpy().flatten()
            labels = labels.cpu().numpy().flatten()

            mae = mean_absolute_error(labels, preds)
            rmse = np.sqrt(mean_squared_error(labels, preds))

            total_mae += mae
            total_rmse += rmse
            sample_count += len(labels)

            all_preds.extend(preds)
            all_labels.extend(labels)
 
    # Print MAE and RMSE for all samples
    print(f'Total MAE Sum: {total_mae:.4f}')
    print(f'Total RMSE Sum: {total_rmse:.4f}')

    # Print real labels and predicted labels
    all_preds = np.array(all_preds).flatten()
    all_labels = np.array(all_labels).flatten()
    print("Real label value:", all_labels)
    print("Predict label values:", all_preds)

    # Save model
    #torch.save(model.state_dict(), "multi_modal_model.pth")
    print("Model saved successfully!")

if __name__ == '__main__':
    from torch.multiprocessing import freeze_support
    freeze_support()
    main()
