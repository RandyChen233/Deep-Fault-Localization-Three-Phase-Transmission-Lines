import scipy.io
import torch
import os
from torch.utils.data import Dataset

# Define custom Dataset class
class FaultDataset(Dataset):
    def __init__(self, VInorm, segmentation_labels, localization_labels):
        self.VInorm = VInorm
        self.segmentation_labels = segmentation_labels
        self.localization_labels = localization_labels
    
    def __len__(self):
        return len(self.VInorm)
    
    def __getitem__(self, idx):
        inputs = self.VInorm[idx]
        segmentation_label = self.segmentation_labels[idx]
        localization_label = self.localization_labels[idx]
        return inputs, segmentation_label, localization_label

def load_and_preprocess_data(file_path, device = 'cpu'):
    """
    Load and preprocess data from a .mat file.

    Args:
        file_path (str): Path to the .mat file.
        device (str): Device to load tensors to ('cpu' or 'cuda').

    Returns:
        dataset (FaultDataset): Preprocessed dataset.
    """
     
    mat = scipy.io.loadmat(file_path)

    # Extract data from the MATLAB structure
    localization_labels = mat['fault_locations'].T 
    fault_signals = mat['Fault_Signals']   
    VInorm = mat['VInorm']           

    # Convert to PyTorch tensors
    fault_signals = torch.tensor(fault_signals, dtype=torch.float32).to(device)
    VInorm = torch.tensor(VInorm, dtype=torch.float32).to(device)
    localization_labels = torch.tensor(localization_labels, dtype=torch.float32).to(device)

    num_samples = localization_labels.shape[1]
    fault_signals = fault_signals.view(num_samples, -1, 2)
    VInorm = VInorm.view(num_samples, -1, 6)
    
    # Assuming segmentation labels are part of fault_signals
    segmentation_labels = fault_signals

    # Create Dataset
    dataset = FaultDataset(VInorm, segmentation_labels, localization_labels.T)

    return dataset

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data_path = 'trainData.mat'
    dataset = load_and_preprocess_data(data_path, device)

    # Save the dataset for future use
    # os.makedirs('data', exist_ok=True)
    # torch.save(dataset, 'data/dataset.pt')
