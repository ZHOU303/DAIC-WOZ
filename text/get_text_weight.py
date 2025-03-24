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
train_dir1 = r'D:\DAIC-WOZ\text_file'  
label_file = r'D:\Dataset\DAIC-WOZ Dataset\train_split_Depression_AVEC2017.csv'

# hyper-parameters
max_sequence_length = 100
batch_size = 24
num_epochs = 200
learning_rate = 0.001

# Read tags
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
            # Limit maximum length
            if features.size(0) > max_sequence_length:
                features = features[:max_sequence_length, :]
            texts.append(features)
            label = label_df.iloc[idx, label_df.columns.get_loc(label_column_name)]
            labels.append(label)
    labels = torch.tensor(labels, dtype=torch.float32).unsqueeze(1).to(device)
    # Fill Alignment
    padded_texts = pad_sequence(texts, batch_first=True, padding_value=0.0).permute(0, 2, 1)
    return padded_texts, labels

texts1, labels1 = load_data(train_dir1, label_df)

# Data Loader
dataset = list(zip(texts1, labels1))
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


# Bi-LSTM + attention
class BiLSTM_Attention(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=224):
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
 
# ResBlock
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
# Define residual network
class ResNet(nn.Module):
    def __init__(self, num_blocks, num_classes=256):
        super(ResNet, self).__init__()
        # Initial convolutional layer
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        # Stacking multiple residual blocks
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
        self.model1 = BiLSTM_Attention(input_dim=512, hidden_dim=128)  # BiLSTM + Attention
        self.fc = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(64, 1),
        ).to(device)

        num_blocks = [2, 2, 2, 2]  # The number of blocks in each stage of ResNet
        self.resnet = ResNet(num_blocks).to(device)  

    def forward(self, x1):
        # Obtain a 224 dimensional vector using the BiLSTM+Attention model
        fused_output = self.model1(x1)  # Output batch_size x 224
        # Calculate the outer product to obtain a matrix of 224x224
        outer_product = torch.bmm(fused_output.unsqueeze(2), fused_output.unsqueeze(1)) 
        outer_product = outer_product.unsqueeze(1)  # Adjust dimensions to(batch_size, 1, 224, 224)

        resnet_output = self.resnet(outer_product)  

        # Fully connected layer for classification output 
        output = self.fc(resnet_output)  # Output the final prediction
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
    for batch_texts1,batch_labels in dataloader:
        batch_texts1,batch_labels = batch_texts1.to(device), batch_labels.to(device)
        optimizer.zero_grad()
        outputs = model(batch_texts1)

        loss = criterion(outputs, batch_labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        # Save the labels and predicted values of the current batch
        true_labels.extend(batch_labels.cpu().numpy())
        predicted_labels.extend(outputs.cpu().detach().numpy())  
    avg_loss = total_loss / len(dataloader)
    # Calculate the MAE and RMSE of the training set
    mae = mean_absolute_error(true_labels, predicted_labels)
    rmse = np.sqrt(mean_squared_error(true_labels, predicted_labels))
    print(f"Epoch [{epoch+1}/{num_epochs}] MAE: {mae:.4f}, RMSE: {rmse:.4f}")
    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save(model.state_dict(), r"D:\DAIC-WOZ\weight\best_text.pth")





