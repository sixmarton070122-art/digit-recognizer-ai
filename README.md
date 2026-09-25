# Handwritten Digit Recognizer

A beginner-friendly project built with Python and PyTorch that trains a convolutional neural network (CNN) to recognize handwritten digits (0–9) using the MNIST dataset.

This project was built as a learning exercise — the goal was to understand PyTorch fundamentals (tensors, datasets, dataloaders, model architecture, training loops, loss functions, optimizers, and evaluation) by implementing everything from scratch rather than using a pre-built solution.

## Current Status

- ✅ Data loading and preprocessing (MNIST via `torchvision`)
- ✅ CNN model architecture
- ✅ Training loop with accuracy tracking
- ✅ Model saving/loading
- ✅ Evaluation on held-out test data (**98.9% test accuracy**)
- 🔜 Drawing GUI — a simple interface to draw a digit and get a live prediction is planned but not yet implemented

## Project Structure

```
.
├── model.py        # CNN architecture (DigitClassifier)
├── main.py         # Trains the model on MNIST and saves weights
├── evaluate.py      # Loads a saved model and reports test accuracy
├── models/          # Saved model weights (.pth files)
└── data/            # MNIST dataset (downloaded automatically)
```

## Model Architecture

A small CNN:

- 2 convolutional layers (16 → 32 channels, 3×3 kernels, padding=1)
- ReLU activations between layers
- 2 fully connected layers narrowing down to 10 output classes (digits 0–9)

## Requirements

- Python 3
- PyTorch
- torchvision

Install dependencies:

```bash
pip install torch torchvision
```

## Usage

### Train the model

```bash
python main.py
```

This downloads MNIST (if not already present), trains the CNN, prints accuracy per epoch, and saves the trained weights to `models/test.pth`.

### Evaluate the saved model

```bash
python evaluate.py
```

This loads the saved weights and reports accuracy on the MNIST test set (data the model never trained on).

## Planned Next Steps

- Build a simple drawing interface (likely using `tkinter`) so a user can draw a digit by hand
- Preprocess the drawn image to match MNIST's format (28×28, grayscale, centered) before feeding it to the model
- Display the predicted digit along with a confidence percentage

## Learning Goals

This project was built step by step to genuinely understand:

- Tensors and how image data is represented
- Datasets and DataLoaders for batching
- Convolutional layers, channels, and how spatial dimensions change through a network
- Training loops: forward pass, loss calculation, backpropagation, optimizer steps
- The difference between training accuracy and test accuracy, and why held-out data matters
- Saving and loading trained models for later use
