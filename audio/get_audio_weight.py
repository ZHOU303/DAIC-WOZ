import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics import mean_absolute_error, mean_squared_error
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# data path
train_dir1 = r'D:\DAIC-WOZ\audio_file\COVAREP' 
train_dir2 = r'D:\DAIC-WOZ\audio_file\FORMANT'  
label_file = r'D:\Dataset\DAIC-WOZ Dataset\train_split_Depression_AVEC2017.csv'

max_sequence_length = 40000  
batch_size = 16
num_epochs = 10
learning_rate = 0.001

label_df = pd.read_csv(label_file)
label_column_name = 'PHQ8_Score'

if label_column_name not in label_df.columns:
    raise ValueError(f"Column not found in tag file: {label_column_name}")

# Load data
def load_data(data_dir, label_df):
    texts, labels = [], []
    csv_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.csv')], key=lambda x: int(x.split('_')[0]))

    for idx, file in enumerate(csv_files):
        file_path = os.path.join(data_dir, file)
        if os.path.exists(file_path):
            data_df = pd.read_csv(file_path)
            features = torch.tensor(data_df.values, dtype=torch.float32).to(device)

            if features.size(0) > max_sequence_length:
                features = features[:max_sequence_length, :]

            texts.append(features)
            
            label = label_df.iloc[idx, label_df.columns.get_loc(label_column_name)]
            labels.append(label)

    labels = torch.tensor(labels, dtype=torch.float32).unsqueeze(1).to(device)
    
    padded_texts = pad_sequence(texts, batch_first=True, padding_value=0.0).permute(0, 2, 1)
    return padded_texts, labels

# Load two sets of data
texts1, labels1 = load_data(train_dir1, label_df)
texts2, labels2 = load_data(train_dir2, label_df)

dataset = list(zip(texts1, texts2, labels1))
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# TCNBlock module
class TCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1):
        super(TCNBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=(kernel_size - 1) * dilation, dilation=dilation)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=(kernel_size - 1) * dilation, dilation=dilation)
        self.norm1 = nn.BatchNorm1d(out_channels)
        self.norm2 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        res = x if self.downsample is None else self.downsample(x)
        x = self.relu(self.norm1(self.conv1(x)))
        x = self.relu(self.norm2(self.conv2(x)))
        x = self.pool(x)  
        return x

# Bi-LSTM + attention
class BiLSTM_Attention(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(BiLSTM_Attention, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.attention = nn.Linear(hidden_dim * 2, 1)  
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x):
        lstm_out, _ = self.lstm(x.permute(0, 2, 1))  
        attention_weights = torch.softmax(self.attention(lstm_out), dim=1)  
        context_vector = torch.sum(attention_weights * lstm_out, dim=1)  
        output = self.fc(context_vector)
        return output

# Intra modal feature fusion
class AttentionFusionLayer(nn.Module):
    def __init__(self, input_dim=224, hidden_dim=448, output_dim=224):
        super(AttentionFusionLayer, self).__init__()
        self.fc1 = nn.Linear(hidden_dim, input_dim)
        self.activation = nn.Tanh()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, feat1, feat2):
        concat1 = torch.cat([feat1, feat2], dim=1) 
        attn_vector = self.activation(self.fc1(concat1))  

        # Fusion features
        result1 = attn_vector * feat1
        result2 = attn_vector * feat2  
        fused_vector = torch.cat([result1, result2], dim=1)
        output = self.fc2(fused_vector)  
        return output
    
# TCN + BiLSTM 
class TCN_BiLSTM_Fusion(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(TCN_BiLSTM_Fusion, self).__init__()
        self.tcn = TCNBlock(input_dim, hidden_dim)
        self.bilstm = BiLSTM_Attention(input_dim, hidden_dim // 2, hidden_dim)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)  

    def forward(self, x):
        tcn_features = self.tcn(x).squeeze(-1)
        bilstm_features = self.bilstm(x)
        fused_features = torch.cat((tcn_features, bilstm_features), dim=1)  
        output = self.fc(fused_features)
        return output
# Define Residual Block (ResBlock)
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlock, self).__init__()
        
        # Define two convolutional layers
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Jump connection (residual connection)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))  
        out = self.bn2(self.conv2(out)) 
        out += self.shortcut(x)  
        out = self.relu(out)  
        return out

class ResNet(nn.Module):
    def __init__(self, num_blocks, num_classes=256):
        super(ResNet, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        
        self.layer1 = self._make_layer(64, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(64, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(128, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(256, 512, num_blocks[3], stride=2)
       
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
      
        self.fc = nn.Linear(512, num_classes)
    
    
    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = []
        layers.append(ResBlock(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(ResBlock(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))  
        x = self.layer1(x)  
        x = self.layer2(x)  
        x = self.layer3(x) 
        x = self.layer4(x)  
        x = self.avgpool(x)  
        x = torch.flatten(x, 1)  
        x = self.fc(x)  
        return x

class FusionModel(nn.Module):
    def __init__(self):
        super(FusionModel, self).__init__()
        self.model1 = TCN_BiLSTM_Fusion(input_dim=74, hidden_dim=112, output_dim=224).to(device)
        self.model2 = TCN_BiLSTM_Fusion(input_dim=5, hidden_dim=112, output_dim=224).to(device)
        self.attention_fusion = AttentionFusionLayer(224, 448, 224).to(device)

        self.fc = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(64, 1),
        ).to(device)

    def forward(self, x1, x2):
        feat1 = self.model1(x1)       
        feat2 = self.model2(x2)       
        fused_output = self.attention_fusion(feat1, feat2)
        # Outer product
        outer_product = torch.bmm(fused_output.unsqueeze(2), fused_output.unsqueeze(1))
        outer_product = outer_product.unsqueeze(1)
       
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        num_blocks = [2, 2, 2, 2]  
        model = ResNet(num_blocks).to(device)
        output = model(outer_product)
      
        output = self.fc(output)  
        return output

# Model initialization and training
model = FusionModel().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
best_loss = float('inf')  

# Training Phase
for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    true_labels = []
    predicted_labels = []

    for batch_texts1, batch_texts2, batch_labels in dataloader:
        batch_texts1, batch_texts2, batch_labels = batch_texts1.to(device), batch_texts2.to(device), batch_labels.to(device)

        optimizer.zero_grad()
        outputs = model(batch_texts1, batch_texts2)

        loss = criterion(outputs, batch_labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        
        true_labels.extend(batch_labels.cpu().numpy())
        predicted_labels.extend(outputs.cpu().detach().numpy())  
    avg_loss = total_loss / len(dataloader)

    # Calculate MAE and RMSE
    mae = mean_absolute_error(true_labels, predicted_labels)
    rmse = np.sqrt(mean_squared_error(true_labels, predicted_labels))

    print(f"Epoch [{epoch+1}/{num_epochs}] MAE: {mae:.4f}, RMSE: {rmse:.4f}")
    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save(model.state_dict(), r"D:\DAIC-WOZ\weight\best_audio.pth")
        



