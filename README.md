# Digit Recognizer

This repository contains the code for a digit recognition model using deep learning. The project is structured into several key components, including data loading, model training, and inference.

This project is a solution for the [Kaggle Digit Recognizer competition](https://www.kaggle.com/competitions/digit-recognizer/overview).

### *Submission Score: 0.99171*

## Repository Structure

- `dataloader.py`: Handles the loading and preprocessing of the digit dataset.
- `DigitRecognizerModel.py`: Defines the architecture of the digit recognition model.
- `inference.py`: Contains the code to perform inference using the trained model.
- `main.py`: The main script to run the training and evaluation processes.
- `trainer.py`: Implements the training loop and evaluation functions.

## Getting Started

### Prerequisites

Make sure you have the following dependencies installed:

- Python 3.7 or higher
- PyTorch
- NumPy

You can install the required packages using pip

## Usage

1. **Data Loading**: Ensure your digit dataset is in the appropriate format and update the paths in `dataloader.py` if necessary.

2. **Training the Model and Perform Inference**: To train the model, run the following command:

    ```bash
    python main.py
    ```

    This will load the data, train the model, and save the trained model to a file and create a `final_submission.csv` for kaggle.

## Files Overview

### `dataloader.py`

This file contains functions to load and preprocess the digit dataset. It includes transformations and data augmentation techniques to improve the model's performance.

### `DigitRecognizerModel.py`

This file defines the neural network architecture for the digit recognizer. It includes the layers, activation functions, and forward pass logic.

### `inference.py`

This script loads a trained model and performs inference on new data. It outputs the predicted digit for each input sample.

### `main.py`

The main script orchestrates the data loading, model training, and evaluation processes. It integrates the components defined in the other scripts and runs the training loop.

### `trainer.py`

This file contains the training loop and evaluation functions. It handles the optimization process, loss calculation, and performance metrics tracking.
