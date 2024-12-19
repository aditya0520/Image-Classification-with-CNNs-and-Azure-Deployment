
# Image Classification with CNNs and Azure Deployment

This repository contains the implementation of an image classification pipeline using Convolutional Neural Networks (CNNs). The project trains, evaluates, and deploys models to classify images from the CIFAR-10 dataset into 10 categories. The best-performing model, AlexNet, is deployed on Azure as a container service for real-time inference.

## Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Training and Evaluation](#training-and-evaluation)
- [Model Deployment](#model-deployment)
- [Usage](#usage)
- [References](#references)

---

## Project Overview

The CIFAR-10 dataset consists of 60,000 32x32 RGB images categorized into 10 classes (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck). This project focuses on:

1. Training CNN architectures (LeNet and AlexNet) using PyTorch.
2. Tuning hyperparameters for improved accuracy.
3. Deploying the best model (AlexNet) to Azure for real-time image classification.

---

## Features

- **CNN Architectures**: Implements LeNet and AlexNet models in PyTorch.
- **Dataset**: Utilizes the CIFAR-10 dataset for training and evaluation.
- **Deployment**: Deploys the best-performing model (AlexNet) to Azure using a containerized service.
- **Inference**: Supports real-time image classification via REST API.

---

## Training and Evaluation

### Training

The models (LeNet and AlexNet) were trained on the CIFAR-10 dataset using the following hyperparameters:
- Optimizer: SGD
- Learning rate: 0.01
- Batch size: 64
- Epochs: 50

### Evaluation

Accuracy scores:
- **LeNet**: Achieved an accuracy of 65% on the test set.
- **AlexNet**: Achieved an accuracy of 71% on the test set.

---

## Model Deployment

The best-performing model (AlexNet) was deployed to Azure using a container service. Deployment steps include:

1. **Model Serialization**: Saved the trained AlexNet model using `torch.save()`.
2. **Scoring Script**: Developed a `score.py` file for loading the model and running inference.
3. **Environment Configuration**: Created an `environment.yml` file to specify dependencies (e.g., PyTorch, Azure SDK).
4. **Local Testing**: Validated the deployment locally using `LocalWebservice`.
5. **Azure Deployment**: Published the model to Azure as a public endpoint using `AciWebService`.

---

## Usage


### REST API Endpoint

After deploying to Azure, use the scoring URI for real-time predictions. Example:
```python
import requests
import json

url = "YOUR_SCORING_URI"
data = {
    "image": [0.1, 0.7, 0.3, ...]  # Flattened 32x32 image array
}
response = requests.post(url, json=data)
print(response.json())
```

---

## References

- CIFAR-10 Dataset: [Krizhevsky](https://www.cs.toronto.edu/~kriz/cifar.html)
- PyTorch Documentation: [PyTorch](https://pytorch.org)
- Azure Machine Learning: [Azure ML](https://azure.microsoft.com/en-us/services/machine-learning/)
