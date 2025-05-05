# Trainer class for training a model

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os

# Directory to save the model
DIR = "output/"

class Trainer:
    def __init__():
        ...
    
    def train(self):
        """
        Train the model on the training set.
        """
        
        # Set the model to training mode
        self.model.train()
        
        # Iterate over the training data
        for batch in self.train_loader:
            # Get the inputs and labels
            inputs, labels = batch
            
            # Move the inputs and labels to the device
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            
            # Zero the gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(inputs)
            
            # Compute the loss
            loss = self.criterion(outputs, labels)
            
            # Backward pass and optimization
            loss.backward()
            self.optimizer.step()

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
            for batch in self.test_loader:
                # Get the inputs
                inputs = batch
                
                # Move the inputs to the device
                inputs = inputs.to(self.device)
                
                # Forward pass
                outputs = self.model(inputs)
                
                # Get the predicted class - multiclass classification
                _, predicted = torch.max(outputs.data, 1)
                
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
            for batch in self.val_loader:
                # Get the inputs and labels
                inputs, labels = batch
                
                # Move the inputs and labels to the device
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                # Forward pass
                outputs = self.model(inputs)
                
                # Compute the loss
                loss = self.criterion(outputs, labels)
                
                # Update the total loss
                total_loss += loss.item()
                
                # Get the predicted class - multiclass classification
                _, predicted = torch.max(outputs.data, 1)
                
                # Update the total and correct counts
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        # Compute the average loss and accuracy
        avg_loss = total_loss / len(self.val_loader)
        accuracy = 100 * correct / total
        
        return avg_loss, accuracy