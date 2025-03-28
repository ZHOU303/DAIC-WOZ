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

        # Text modal processing with Convolutional Layers 
        self.text_conv1 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.text_bn1 = nn.BatchNorm2d(128)
        self.text_conv2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.text_bn2 = nn.BatchNorm2d(128)
        self.text_proj = nn.Conv2d(256, 128, kernel_size=1)

        # Audio modal processing with Convolutional Layers 
        self.audio_conv1 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.audio_bn1 = nn.BatchNorm2d(128)
        self.audio_conv2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.audio_bn2 = nn.BatchNorm2d(128)
        self.audio_proj = nn.Conv2d(256, 128, kernel_size=1)

        # Video modal processing with Convolutional Layers 
        self.video_conv1 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.video_bn1 = nn.BatchNorm2d(128)
        self.video_conv2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.video_bn2 = nn.BatchNorm2d(128)
        self.video_proj = nn.Conv2d(256, 128, kernel_size=1)

        # Learnable weights for skip connections
        self.alpha = nn.Parameter(torch.tensor(0.5))  # Learnable parameter for fusion: Text + Video
        self.beta = nn.Parameter(torch.tensor(0.5))   # Learnable parameter for fusion: Text + Audio

        # Fusion of text (with residual video) and audio features
        self.fusion_fc = nn.Linear(301056, 128) 
        # Final prediction layer
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 1)

        # Dropout layer (dropout probability set to 0.3)
        self.dropout = nn.Dropout(0.3)

    def forward(self, text, audio, video):
    
        res_text = F.relu(self.text_bn1(self.text_conv1(text)))
        res_text = self.text_bn2(self.text_conv2(res_text)) + self.text_proj(text)  
        res_text = F.relu(res_text)

      
        res_video = F.relu(self.video_bn1(self.video_conv1(video)))
        res_video = self.video_bn2(self.video_conv2(res_video)) + self.video_proj(video)  
        res_video = F.relu(res_video)

        res_text = self.alpha * res_video + (1 - self.alpha) * res_text  

        res_audio = F.relu(self.audio_bn1(self.audio_conv1(audio)))  
        res_audio = self.audio_bn2(self.audio_conv2(res_audio)) + self.audio_proj(audio)  
        res_audio = F.relu(res_audio)

        res_text = self.beta * res_audio + (1 - self.beta) * res_text  

        # Flatten all features before passing to fully connected layer
        res_text = res_text.view(res_text.size(0), -1)  

        # Fusion of text+video+audio features
        fusion = torch.cat((res_audio.view(res_audio.size(0), -1), res_text, res_video.view(res_video.size(0), -1)), dim=1)  
        fusion = F.relu(self.fusion_fc(fusion))

        # Final classification
        out = F.relu(self.fc1(fusion))
        out = self.fc2(out)
        return out

def main():
    text_dir = r'D:\DAIC-WOZ\parameter\text'
    audio_dir = r'D:\DAIC-WOZ\parameter\audio'
    video_dir = r'D:\DAIC-WOZ\parameter\visual'
    label_file = r'D:\DAIC-WOZ\train_split_Depression_AVEC2017.csv'

    batch_size = 24
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
