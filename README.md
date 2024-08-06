# Deep RCNN network for localizing LG fault in a 3-phase high voltage power transmission line.

## Overview

This repository contains the implementation and visualization of a Recurrent Convolutional Neural Network (RCNN) designed for fault localization in 3-phase transmission lines. The model leverages convolutional layers for feature extraction, LSTM layers for capturing temporal dependencies, and fully connected layers for final output generation, including fault segmentation identification and fault localization tasks.

## Table of Contents

- [Overview](#overview)
- [Model Architecture](#model-architecture)
- [Setup](#setup)
- [Usage](#usage)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Model Architecture

### Summary

| Layer Type            | Parameters                                                                                         | Details                                                                                         |
|-----------------------|----------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------|
| **Input**             | Input Channels: 6                                                                                  | 3-phase voltage and 3-phase current                                                             |
| **Conv1**             | Type: 1D Convolution<br>Output Channels: 16<br>Kernel Size: 3<br>Activation: LeakyReLU (slope: 0.1) |                                                                                                 |
| **Conv2**             | Type: 1D Convolution<br>Output Channels: 32 (16 * 2)<br>Kernel Size: 3<br>Activation: LeakyReLU (slope: 0.1) |                                                                                                 |
| **Conv3**             | Type: 1D Convolution<br>Output Channels: 64 (32 * 2)<br>Kernel Size: 3<br>Activation: LeakyReLU (slope: 0.1) |                                                                                                 |
| **Pooling Layer**     | Type: 1D Max Pooling                                                                              | Applied after the third convolutional layer                                                     |
| **Dropout Layer**     | Dropout Probability: 0.5                                                                          | Applied after the pooling layer                                                                 |
| **LSTM Layer**        | Type: LSTM (Long Short-Term Memory)<br>Input Size: 64<br>Hidden Size: 128<br>Number of Layers: 2<br>Batch First: True |                                                                                                 |
| **Fully Connected Layer 1 (FC1)** | Type: Fully Connected (Linear)<br>Input Size: 128<br>Output Size: 128<br>Activation: ReLU      |                                                                                                 |
| **Segmentation Head** | Type: Fully Connected (Linear)<br>Input Size: 128<br>Output Size: 40002 (seq_length * 2)<br>Reshape to: (batch_size, seq_length, 2) |                                                                                                 |
| **Localization Head** | Type: Fully Connected (Linear)<br>Input Size: 128<br>Output Size: 2                                 |                                                                                                 |

### Visual Representation

You can visualize the RCNN architecture by compiling the provided LaTeX script located in the `latex` directory.

## Setup

### Prerequisites

- Python 3.x
- PyTorch
- NumPy
- Matplotlib

### Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/RCNN-Fault-Localization.git
    cd RCNN-Fault-Localization
    ```

2. Create a virtual environment and activate it:
    ```bash
    python -m venv env
    source env/bin/activate  # On Windows, use `env\Scripts\activate`
    ```

3. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Training the Model

1. Prepare your dataset and update the data paths in `train.py`.
2. Run the training script:
    ```bash
    python train.py
    ```

### Evaluating the Model

1. After training, evaluate the model using the test dataset:
    ```bash
    python test.py
    ```


## Contributing

Contributions are welcome! Please fork the repository and submit a pull request for any enhancements or bug fixes.

### Guidelines

1. Fork the repository.
2. Create a new branch (`git checkout -b feature/your-feature`).
3. Commit your changes (`git commit -m 'Add some feature'`).
4. Push to the branch (`git push origin feature/your-feature`).
5. Open a pull request.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

