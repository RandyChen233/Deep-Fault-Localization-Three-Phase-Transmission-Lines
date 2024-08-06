import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
from model import FaultLocalizationModel
from preprocess import load_and_preprocess_data
from plot import initialize_plot, update_plot, finalize_plot
import torch.nn as nn
import argparse
import random
import numpy as np
import os

def save_datasets(train_dataset, val_dataset, test_dataset, directory='datasets'):
    os.makedirs(directory, exist_ok=True)
    torch.save(train_dataset, os.path.join(directory, 'train_dataset.pt'))
    torch.save(val_dataset, os.path.join(directory, 'val_dataset.pt'))
    torch.save(test_dataset, os.path.join(directory, 'test_dataset.pt'))

def train_model(model, 
                train_loader, 
                val_loader, 
                optimizer, 
                criterion_segmentation, 
                criterion_localization, 
                num_epochs=10, 
                device = 'cpu'):
    
    # Initialize lists to store loss values
    train_losses = []
    val_losses_segmentation = []
    val_losses_localization = []

    # Initialize minimum loss trackers
    min_loss_v = float('inf')
    min_loss_t = float('inf')

    # Initialize plot
    fig, ax, train_loss_line, val_loss_segmentation_line, val_loss_localization_line = initialize_plot(num_epochs)

    segmentation_weight = 1000
    localization_weight = 0.005
    check_interval = 5

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels_segmentation, labels_localization in train_loader:
            inputs, labels_segmentation, labels_localization = inputs.to(device), labels_segmentation.to(device), labels_localization.to(device)
            optimizer.zero_grad()
            
            # Forward pass
            outputs_segmentation, outputs_localization = model(inputs.permute([0,2,1]))
            
            # Compute losses
            loss_segmentation = criterion_segmentation(outputs_segmentation, labels_segmentation)
            loss_localization = criterion_localization(outputs_localization, labels_localization)
            
            # Total loss
            loss = segmentation_weight * loss_segmentation + localization_weight * loss_localization
            loss.backward()
            # Print gradient norms before clipping
            total_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5
            print(f'Epoch [{epoch+1}/{num_epochs}], Gradient Norm: {total_norm:.4f}')

            # Clip gradients to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=100)
            
            optimizer.step()
            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Evaluation on validation set
        model.eval()
        with torch.no_grad():
            val_loss_segmentation = 0
            val_loss_localization = 0
            
            for inputs, labels_segmentation, labels_localization in val_loader:
                inputs, labels_segmentation, labels_localization = inputs.to(device), labels_segmentation.to(device), labels_localization.to(device)
                
                outputs_segmentation, outputs_localization = model(inputs.permute([0,2,1]))
                val_loss_segmentation += segmentation_weight * criterion_segmentation(outputs_segmentation, labels_segmentation).item()
                val_loss_localization += localization_weight * criterion_localization(outputs_localization, labels_localization).item()
        
        avg_val_loss_segmentation = val_loss_segmentation / len(val_loader)
        avg_val_loss_localization = val_loss_localization / len(val_loader)
        avg_val_loss = avg_val_loss_segmentation + avg_val_loss_localization
        val_losses_segmentation.append(avg_val_loss_segmentation)
        val_losses_localization.append(avg_val_loss_localization)
        
        # Save model if validation loss decreases
        if avg_val_loss < min_loss_v and epoch % check_interval == 0:
            min_loss_v = avg_val_loss
            print(f'Save model at epoch {epoch+1}, mean of valid loss: {avg_val_loss}')
            torch.save(model.state_dict(), 'trained_models/model.valid.pth')
            torch.save(optimizer.state_dict(), 'trained_models/optimizer.valid.pth')
        
        # Save model if training loss decreases
        if avg_train_loss < min_loss_t and epoch % check_interval == 0:
            min_loss_t = avg_train_loss
            print(f'Save model at epoch {epoch+1}, mean of train loss: {avg_train_loss}')
            torch.save(model.state_dict(), 'trained_models/model.train.pth')
            torch.save(optimizer.state_dict(), 'trained_models/optimizer.train.pth')
        
        # Update plot
        update_plot(fig, 
                    ax, 
                    train_loss_line, 
                    val_loss_segmentation_line, 
                    val_loss_localization_line, 
                    epoch, train_losses, 
                    val_losses_segmentation, 
                    val_losses_localization)

        print(f'Epoch {epoch+1}/{num_epochs}, '
            f'Train Loss: {avg_train_loss}, '
            f'Val Loss (Segmentation): {avg_val_loss_segmentation}, '
            f'Val Loss (Localization): {avg_val_loss_localization}')

    # Finalize and save plot
    finalize_plot(fig)


if __name__ == "__main__":

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_path = 'trainData.mat'
    dataset = load_and_preprocess_data(data_path)

    # Perform train-validation-test split
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    save_datasets(train_dataset, val_dataset, test_dataset)

    # Define model and optimizer
    input_channels = 6  # 3-phase voltage and 3-phase current
    cnn_out_channels = 16
    cnn_kernel_size = 3
    lstm_hidden_size = 128
    lstm_layers = 2
    fc_size = 128
    seq_length = 20001
    num_epochs = 400

    model = FaultLocalizationModel(input_channels, cnn_out_channels, cnn_kernel_size, lstm_hidden_size, lstm_layers, fc_size, seq_length).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0005)

    criterion_segmentation = nn.BCEWithLogitsLoss()  # Binary cross-entropy for segmentation
    criterion_localization = nn.MSELoss()  # Mean squared error for localization

    train_model(model, train_loader, val_loader, optimizer, criterion_segmentation, criterion_localization, num_epochs, device)
