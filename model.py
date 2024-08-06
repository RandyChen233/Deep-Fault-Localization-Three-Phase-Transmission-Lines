import torch
import torch.nn as nn
import torch.nn.functional as F

# Define the model
class FaultLocalizationModel(nn.Module):
    def __init__(self, input_channels, cnn_out_channels, cnn_kernel_size, lstm_hidden_size, lstm_layers, fc_size, seq_length):
        super(FaultLocalizationModel, self).__init__()
        
        # Convolutional layers for feature extraction
        self.conv1 = nn.Conv1d(input_channels, cnn_out_channels, kernel_size=cnn_kernel_size)
        self.conv2 = nn.Conv1d(cnn_out_channels, cnn_out_channels*2, kernel_size=cnn_kernel_size)
        self.conv3 = nn.Conv1d(cnn_out_channels*2, cnn_out_channels*4, kernel_size=cnn_kernel_size)
        self.pool = nn.MaxPool1d(2)
        # Dropout layer
        self.dropout = nn.Dropout(p=0.5)

        # LSTM for temporal analysis
        self.lstm = nn.LSTM(cnn_out_channels*4, lstm_hidden_size, lstm_layers, batch_first=True)
        
        # Fully connected layers for Segmentation Head
        self.fc1 = nn.Linear(lstm_hidden_size, fc_size)
        self.fc_segmentation = nn.Linear(fc_size, seq_length*2)  # Output shape: (batch_size, seq_length*2)
        
        # Fully connected layers for Localization Head
        self.fc_localization = nn.Linear(fc_size, 2)  # Output shape: (batch_size, 2)

    def forward(self, x):
        # Convolutional layers
        x = F.leaky_relu(self.conv1(x), negative_slope=0.1)
        x = F.leaky_relu(self.conv2(x), negative_slope=0.1)
        x = self.pool(F.leaky_relu(self.conv3(x), negative_slope=0.1))

        # Apply dropout
        x = self.dropout(x)

        # Prepare data for LSTM
        x = x.permute(0, 2, 1)  # (batch_size, seq_length, features)
        
        # LSTM layers
        x, _ = self.lstm(x)
        x = x[:, -1, :]  # Take the output from the last time step
        
        # Fully connected layers for segmentation and localization
        x = F.relu(self.fc1(x))
        segmentation_out = self.fc_segmentation(x).view(x.size(0), -1, 2)  # Reshape to (batch_size, seq_length, 2)
        localization_out = self.fc_localization(x)
        
        return segmentation_out, localization_out