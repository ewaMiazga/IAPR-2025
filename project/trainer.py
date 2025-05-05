# Trainer class for training a model

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os
from models.cnn import SimpleCNN

# Directory to save the model
DIR = "output/"

class Trainer:
    def __init__(self, model=None, loss_fn=None, optimizer=None,
                 train_loader=None, val_loader=None, test_loader=None):
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
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = self.get_device()
        self.model.to(self.device)
    
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
                probs = torch.sigmoid(outputs)
                predicted = (probs > 0.5).float()
                
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
                probs = torch.sigmoid(outputs)
                predicted = (probs > 0.5).float()
                
                # Update the total and correct counts
                correct += (predicted == labels.int()).all(dim=1).sum().item()
                total += labels.size(0)
        
        # Compute the average loss and accuracy
        avg_loss = total_loss / len(self.val_loader)
        accuracy = 100 * correct / total
        
        return avg_loss, accuracy
    
if __name__ == "__main__":
    # Example usage
    trainer = Trainer()

    # CNN model
    model = SimpleCNN(input_shape=3, hidden_units=64, output_shape=10)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(params=model.parameters(),
                             lr=0.1)

    trainer = Trainer(model=model,
                  loss_fn=loss_fn,
                  optimizer=optimizer,
                  train_loader=train_loader,
                  val_loader=val_loader,
                  test_loader=test_loader)
    
    trainer.train()
    predictions = trainer.predict()
    print(predictions)
    trainer.save_model()
    avg_loss, accuracy = trainer.evaluate()
    print(f"Average Loss: {avg_loss}, Accuracy: {accuracy}%")