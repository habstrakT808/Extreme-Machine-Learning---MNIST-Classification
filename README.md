# Extreme Learning Machine (ELM) with Nguyen-Widrow Method

This repository contains an implementation of the **Extreme Learning Machine (ELM)** model using the Nguyen-Widrow initialization method. The project demonstrates ELM's capabilities in classifying the MNIST dataset.

## Overview

ELM is a feedforward neural network with a single hidden layer. It is characterized by its fast learning speed and good generalization performance. This implementation uses:
- Nguyen-Widrow initialization to set weights for the hidden layer.
- MNIST dataset to evaluate the model's performance.

## Features
- ELM model training with Nguyen-Widrow initialization.
- Predictions using the trained ELM model.
- Support for classification tasks.
- Performance evaluation using accuracy and Mean Squared Error (MSE).

## Dataset
The MNIST dataset is used in this project, consisting of:
- 70,000 grayscale images of handwritten digits (28x28 pixels each).
- Labels ranging from 0 to 9.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/habstrakT808/Extreme-Machine-Learning---MNIST-Classification.git
   cd elm-nguyen-widrow
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
### Step 1: Download the Dataset
The dataset is automatically downloaded from Google Drive using the `gdown` library. Ensure that the dataset URL is accessible.

### Step 2: Run the Code
Run the Python script to train and evaluate the model:
```bash
python elm_nguyen_widrow.py
```

### Step 3: Visualize Results
The script includes code to visualize a subset of the dataset and outputs performance metrics such as accuracy and MSE.

## Code Structure
- `elm_nguyen_widrow.py`: Main script implementing the ELM model and evaluating it on the MNIST dataset.
- `data.csv`: The MNIST dataset (downloaded automatically).

## Key Functions
### `elm_fit`
Trains the ELM model using Nguyen-Widrow initialization.
```python
def elm_fit(X, target, h, W=None):
```
- `X`: Input data.
- `target`: Target labels (one-hot encoded).
- `h`: Number of neurons in the hidden layer.
- `W`: (Optional) Predefined weights.

### `elm_predict`
Generates predictions using the trained ELM model.
```python
def elm_predict(X, W, beta, round_output=False):
```
- `X`: Input data.
- `W`: Input-to-hidden layer weights.
- `beta`: Hidden-to-output layer weights.
- `round_output`: Whether to round predictions.

## Results
### Training Metrics
- **MSE**: ~0.0045
- **Accuracy**: ~87%

### Example Visualization
A sample of the MNIST dataset:

![ELM 1](https://github.com/user-attachments/assets/fc1be39a-0b5d-4dab-8fe7-8ce09886cde5)

![ELM](https://github.com/user-attachments/assets/ef2bb70a-9ae0-4258-b5c6-89ca0f5a2c16)

## Future Improvements
- Explore other activation functions.
- Experiment with different weight initialization methods.
- Apply ELM to other datasets for broader evaluation.

## Requirements
- Python 3.8+
- NumPy
- Pandas
- Matplotlib
- scikit-learn
- gdown

Install dependencies using:
```bash
pip install -r requirements.txt
```

## References
- [Nguyen-Widrow Initialization Method](https://link-to-paper-or-explanation)
- [Extreme Learning Machine (ELM)](https://link-to-elm-overview)
- [MNIST Dataset](http://yann.lecun.com/exdb/mnist/)

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

## Author
Hafiyan Al Muqaffi Umary

For any questions or suggestions, feel free to [contact me](mailto:jhodywiraputra@gmail.com).
