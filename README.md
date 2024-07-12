# PRODIGY_ML_05



# Food-101 Classification with DenseNet201

This project demonstrates the process of training a DenseNet201 model on the Food-101 dataset to classify images of various food items.

## Table of Contents
- [Dataset](#dataset)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Evaluation](#evaluation)
- [Acknowledgements](#acknowledgements)

## Dataset
The Food-101 dataset contains 101 food categories with 101,000 images. For the purposes of this example, we use a subset of 20 classes plus one 'other' class.

Dataset source: [Food-101 Dataset](https://www.vision.ee.ethz.ch/datasets_extra/food-101/)

## Requirements
- Python 3.6+
- PyTorch
- torchvision
- pandas
- numpy
- tqdm
- PIL (Pillow)
- requests
- scikit-learn

## Installation
Clone the repository and navigate to the project directory:

```bash
git clone https://github.com/your-username/food-classification.git
cd food-classification
```

Install the required packages:

```bash
pip install torch torchvision pandas numpy tqdm pillow requests scikit-learn
```

## Usage

### Download and Extract the Dataset
The code automatically downloads and extracts the Food-101 dataset. Ensure you have a stable internet connection for this step.

### Preparing the Data
The dataset is processed to create training and testing DataFrames containing image paths and labels. Data augmentation techniques are applied to enhance the training process.

### Running the Code
Execute the code to train the model:

```python
python train.py
```

### Customizing the Code
You can customize the code by modifying the following sections:

- `train_transforms` and `test_transforms`: Modify the data augmentation and preprocessing techniques.
- `Label_encoder`: Adjust the label encoding process if using different classes.
- `Food20` class: Customize the dataset handling as needed.
- Hyperparameters: Change the learning rate, batch size, and number of epochs to experiment with different settings.

## Model Architecture
The model is based on the DenseNet201 architecture, pre-trained on ImageNet. The classifier is modified to suit the Food-101 dataset.

```python
weights = models.DenseNet201_Weights.IMAGENET1K_V1
model = models.densenet201(weights=weights)
for param in model.parameters():
    param.requires_grad = False
classifier = nn.Sequential(
    nn.Linear(1920, 1024),
    nn.LeakyReLU(),
    nn.Linear(1024, 101),
)
model.classifier = classifier
```

## Training
The training process involves the following steps:

1. Forward pass
2. Calculate loss
3. Backpropagation
4. Optimizer step

```python
def train_step(model, dataloader, loss_fn, optimizer, device):
    model.train()
    train_loss, train_acc = 0, 0
    for batch, (X, y) in enumerate(tqdm(dataloader)):
        images, labels = X.to(device), y.to(device)
        y_pred = model(images)
        loss = loss_fn(y_pred, labels)
        train_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += (y_pred_class == labels).sum().item() / len(y_pred)
    return train_loss / len(dataloader), train_acc / len(dataloader)
```

## Evaluation
The evaluation step calculates the loss and accuracy on the test dataset.

```python
def test_step(model, dataloader, loss_fn, device):
    model.eval()
    test_loss, test_acc = 0, 0
    with torch.inference_mode():
        for batch, (X, y) in enumerate(tqdm(dataloader)):
            images, labels = X.to(device), y.to(device)
            test_pred_logits = model(images)
            loss = loss_fn(test_pred_logits, labels)
            test_loss += loss.item()
            test_pred_labels = torch.argmax(torch.softmax(test_pred_logits, dim=1), dim=1)
            test_acc += ((test_pred_labels == labels).sum().item() / len(test_pred_labels))
    return test_loss / len(dataloader), test_acc / len(dataloader)
```

## Acknowledgements
- The [Food-101 dataset](https://www.vision.ee.ethz.ch/datasets_extra/food-101/) by Lukas Bossard, Matthieu Guillaumin, and Luc Van Gool.
- The PyTorch and torchvision teams for their amazing libraries.

