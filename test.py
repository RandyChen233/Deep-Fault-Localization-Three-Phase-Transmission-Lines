import torch
from torch.utils.data import DataLoader
from model import FaultLocalizationModel
from preprocess import load_and_preprocess_data
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import os 
import csv
import numpy as np

def load_datasets(directory='datasets'):
    train_dataset = torch.load(os.path.join(directory, 'train_dataset.pt'))
    val_dataset = torch.load(os.path.join(directory, 'val_dataset.pt'))
    test_dataset = torch.load(os.path.join(directory, 'test_dataset.pt'))
    return train_dataset, val_dataset, test_dataset

def evaluate_model(model, test_loader, criterion_segmentation, criterion_localization, device='cpu'):
    model.eval()
    segmentation_weight = 1000
    localization_weight = 0.005

    with torch.no_grad():
        test_loss_segmentation = 0
        test_loss_localization = 0

        i = 0
        for inputs, labels_segmentation, labels_localization in test_loader:
            print(f'Inputs has shape {inputs.shape}, fault location has shape {labels_segmentation.shape}')
            inputs, labels_segmentation, labels_localization = inputs.to(device), labels_segmentation.to(device), labels_localization.to(device)
            outputs_segmentation, outputs_localization = model(inputs.permute([0,2,1]))

            # Apply sigmoid to outputs_segmentation to get probabilities
            # Threshold probabilities to get binary predictions
            outputs_segmentation_binary = (torch.sigmoid(outputs_segmentation) > 0.5).float()
            
            test_loss_segmentation += segmentation_weight * criterion_segmentation(outputs_segmentation, labels_segmentation).item()
            test_loss_localization += localization_weight * criterion_localization(outputs_localization, labels_localization).item()

            np.savez(f'results/predict_Results_batch={i}.npz',
                    predictedSegment = outputs_segmentation_binary.cpu(), 
                    trueSegment = labels_segmentation.cpu(),
                    predictedLocation = outputs_localization.cpu(),
                    trueLocation = labels_localization.cpu())
            
            i += 1

    avg_test_loss_segmentation = test_loss_segmentation / len(test_loader)
    avg_test_loss_localization = test_loss_localization / len(test_loader)
    print(f'Test Loss (Segmentation): {avg_test_loss_segmentation}, Test Loss (Localization): {avg_test_loss_localization}')
    # print(f'Predicted fault segment is {outputs_segmentation_binary[-1]}, true fault segment is {labels_segmentation[-1]}')


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    _, _, test_dataset = load_datasets()
  
    # Create DataLoader for test set
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    # Load the trained model
    input_channels = 6  # 3-phase voltage and 3-phase current
    cnn_out_channels = 16
    cnn_kernel_size = 3
    lstm_hidden_size = 128
    lstm_layers = 2
    fc_size = 128
    seq_length = 20001

    model = FaultLocalizationModel(input_channels, 
                                   cnn_out_channels, 
                                   cnn_kernel_size, 
                                   lstm_hidden_size, 
                                   lstm_layers, 
                                   fc_size, 
                                   seq_length).to(device)
    
    model.load_state_dict(torch.load('trained_models/model.train.pth'))

    criterion_segmentation = nn.BCEWithLogitsLoss()  # Binary cross-entropy for segmentation
    criterion_localization = nn.MSELoss()  # Mean squared error for localization

    evaluate_model(model, test_loader, criterion_segmentation, criterion_localization, device)
