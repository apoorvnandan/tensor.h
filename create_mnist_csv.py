import torch
import torchvision
import torchvision.transforms as transforms
import csv
import numpy as np
from torch.utils.data import DataLoader

# Assuming you have already defined the transforms as in your snippet
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

# Load datasets
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

def dataset_to_csv(dataset, filename):
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)

        # Write header - 784 columns for pixels + 10 for one-hot encoded label
        # header = [f'pixel{i}' for i in range(28*28)] + [f'label_{j}' for j in range(10)]
        # writer.writerow(header)

        for image, label in dataset:
            # Flatten the image
            image_flat = image.numpy().flatten()
            # One-hot encode the label
            label_one_hot = np.zeros(10)
            label_one_hot[label] = 1

            # Combine image data with one-hot encoded label
            row = np.concatenate([image_flat, label_one_hot])
            writer.writerow(row)

# Convert trainset to CSV
dataset_to_csv(trainset, 'mnist_train.csv')

# Convert testset to CSV
dataset_to_csv(testset, 'mnist_test.csv')

print("Datasets have been converted to CSV format.")
