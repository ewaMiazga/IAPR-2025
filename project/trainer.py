# Trainer class for training a model

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os
from models.cnn import SimpleCNN
import numpy as np
from sklearn.metrics import f1_score

# Directory to save the model
DIR = "output/"

class Trainer:
    def __init__(self, model=None, loss_fn=None, optimizer=None, scheduler=None, num_epochs=None,
                 train_loader=None, val_loader=None, test_loader=None, device='cpu'):
        """
        Initialize the Trainer class.
        Parameters:
        - model (torch.nn.Module): The model to be trained.
        - loss_fn (torch.nn.Module): The loss function to be used.
        - optimizer (torch.optim.Optimizer): The optimizer to be used.
        - train_loader (torch.utils.data.DataLoader): DataLoader for the training set.
        - val_loader (torch.utils.data.DataLoader): DataLoader for the validation set.
        - test_loader (torch.utils.data.DataLoader): DataLoader for the test set.
        """
        self.model = model
        self.criterion = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.num_epochs = num_epochs
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.model.to(self.device)
    
    def train(self):
        """
        Train the model on the training set.
        """
        
        # Set the model to training mode
        self.model.train()
        val_losses = []
        val_metrics = []
        
        for epoch in range(self.num_epochs):
            epoch_loss = 0.0
            # Iterate over the training data
            for images, targets in self.train_loader:

                ## coco
                #inputs = torch.stack(images).to(self.device)

                ## Extract the first label from each target dict
                #labels = torch.tensor([t['labels'][0].item() - 1 for t in targets], dtype=torch.long).to(self.device)

                ## patches 
                inputs = images.to(self.device)          # Already stacked tensors
                labels = targets.to(self.device)          # Already tensor of shape [batch_size]
                
                # Zero the gradients
                self.optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(inputs)
                
                # Compute the loss
                loss = self.criterion(outputs, labels)
                
                # Backward pass and optimization
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()

            avg_epoch_loss = epoch_loss / len(self.train_loader)
            print(f"Epoch [{epoch+1}/{self.num_epochs}], Loss: {avg_epoch_loss:.4f}")

            # === Validation ===
            val_loss, val_metric = self.evaluate()  # accuracy or MAE or whatever
            val_losses.append(val_loss)
            val_metrics.append(val_metric)

            # === Scheduler step ===
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)  # based on validation loss
                else:
                    self.scheduler.step()

    def predict(self):
        """
        Make predictions using the trained model.
        """
        # Set the model to evaluation mode
        self.model.eval()
        
        # Initialize an empty list to store predictions
        predictions = []
        
        # Disable gradient calculation for inference
        with torch.no_grad():
            # Iterate over the test data
            for images in self.test_loader:
                # Get the inputs
                ## coco 
                #inputs = torch.stack(images).to(self.device)

                ## patches
                inputs = images.to(self.device)          # Already stacked tensors


                # Move the inputs to the device
                #inputs = inputs.to(self.device)
                
                # Forward pass
                outputs = self.model(inputs)
                
                _, predicted = torch.max(outputs, dim=1)
                
                # Append the predictions to the list
                predictions.append(predicted.cpu().numpy())
        
        return predictions

    def save_model(self):
        """
        Save the model to the specified directory.
        """
        if not os.path.exists(DIR):
            os.makedirs(DIR)
        torch.save(self.model.state_dict(), os.path.join(DIR, 'model.pth'))

    def load_model(self):
        """
        Load the model from the specified directory.
        """
        if os.path.exists(os.path.join(DIR, 'model.pth')):
            self.model.load_state_dict(torch.load(os.path.join(DIR, 'model.pth')))
        else:
            raise FileNotFoundError("Model file not found.")
        

    def evaluate(self):
        """
        Evaluate the model on the validation set.
        """
        # Set the model to evaluation mode
        self.model.eval()
        
        # Initialize variables to track the total loss and accuracy
        total_loss = 0.0
        correct = 0
        total = 0
        
        # Disable gradient calculation for evaluation
        with torch.no_grad():
            # Iterate over the validation data
            for images, targets in self.val_loader:

                ## coco
                # inputs = torch.stack(images).to(self.device)

                # # Extract the first label from each target dict
                # labels = torch.tensor([t['labels'][0].item() - 1 for t in targets], dtype=torch.long).to(self.device)
                
                ## patches
                inputs = images.to(self.device)          # Already stacked tensors
                labels = targets.to(self.device)          # Already tensor of shape [batch_size]
                
                # Forward pass
                outputs = self.model(inputs)
                
                # Compute the loss
                loss = self.criterion(outputs, labels)
                
                # Update the total loss
                total_loss += loss.item()
                
                # Get the predicted class - multiclass classification
                _, predicted = torch.max(outputs, dim=1)
                
                # Update the total and correct counts
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
        
        # Compute the average loss and accuracy
        avg_loss = total_loss / len(self.val_loader)
        accuracy = 100 * correct / total
        
        return avg_loss, accuracy
    
    