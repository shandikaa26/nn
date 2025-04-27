# Neural Network for Water Potability Classification

A simple neural network implementation in Rust that predicts water potability using a three-layer neural network architecture.

## Features

- Implements a feedforward neural network with three hidden layers
- Uses ReLU activation for hidden layers and sigmoid for output
- Includes data normalization and shuffling
- Visualizes training accuracy with plotters
- Displays progress with indicatif progress bar

## Dataset

This project uses the Water Potability dataset, which contains water quality metrics for determining if water is safe to drink. The dataset has 9 input features and a binary output (potable or not potable).

## Requirements

- Rust 1.54+
- Dependencies:
  - ndarray
  - ndarray-rand
  - rand
  - rand_distr
  - csv
  - plotters
  - indicatif

## Usage

1. Place the dataset at `src/water_potability.csv`
2. Run the program:

```
cargo run --release
```

## Model Architecture

- Input layer: 9 features
- First hidden layer: 32 neurons with ReLU activation
- Second hidden layer: 32 neurons with ReLU activation
- Output layer: 1 neuron with Sigmoid activation

## Training Parameters

- Epochs: 2000
- Learning rate: 0.5
- Hidden layer size: 32 neurons

## Output

The program generates:
- Terminal output showing training progress and final accuracy
- An `accuracy.png` visualization of training accuracy over epochs
