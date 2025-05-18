# Trainer class for training a model
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from models.cnn import SimpleCNN
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
from tqdm import tqdm
from PIL import Image


# Directory to save the model
DIR = "output/"

class Trainer:
    def __init__(self, model=None, model_name=None, loss_fn=None, optimizer=None, scheduler=None,
                 num_epochs=None, train_loader=None, val_loader=None, test_loader=None, device='cpu'):
        """
        Initialize the Trainer class.
        Parameters:
        - model (torch.nn.Module): The model to be trained.
        - model_name (str): Name of the model.
        - loss_fn (torch.nn.Module): The loss function to be used.
        - optimizer (torch.optim.Optimizer): The optimizer to be used.
        - scheduler (torch.optim.lr_scheduler): The learning rate scheduler.
        - momentum (float): Momentum for the optimizer.
        - weight_decay (float): Weight decay for the optimizer.
        - train_loader (torch.utils.data.DataLoader): DataLoader for the training set.
        - val_loader (torch.utils.data.DataLoader): DataLoader for the validation set.
        - test_loader (torch.utils.data.DataLoader): DataLoader for the test set.
        """
        self.model = model
        self.model_name = model_name
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
        val_overall_f1 = []
        val_f1_per_class = []
        val_overall_accuracies = []
        val_losses = []
        
        for epoch in range(self.num_epochs):
            epoch_loss = 0.0

            for images, labels in tqdm(self.train_loader, desc=f"Epoch {epoch+1}"):

                if self.model_name == "SimpleCNN":
                    ## patches 
                    inputs = images.to(self.device)          # Already stacked tensors
                    labels = labels.to(self.device)          # Already tensor of shape [batch_size]

                    # Forward pass
                    outputs = self.model(inputs)
                    
                    # Compute the loss
                    loss = self.criterion(outputs, labels)
                
                if self.model_name == "MobileNetV3":
                    images = [img.to(self.device) for img in images]
                    labels = [{k: v.to(self.device) for k, v in t.items()} for t in labels]    

                    # Forward pass and compute loss
                    loss_dict = self.model(images, labels)
                    loss = sum(loss_i for loss_i in loss_dict.values())

                elif self.model_name == "SSDLiteMobileNetV3":
                    images = list(img.to(self.device) for img in images)
                    labels = [{k: v.to(self.device) for k, v in t.items()} for t in labels]

                    loss_dict = self.model(images, labels)
                    loss = sum(loss_i for loss_i in loss_dict.values())
                
                # Zero the gradients
                self.optimizer.zero_grad()

                # Backward pass and optimization
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()

            avg_epoch_loss = epoch_loss / len(self.train_loader)
            print(f"Epoch [{epoch+1}/{self.num_epochs}], Loss: {avg_epoch_loss:.4f}")

            # === Validation ===
            val_loss, overall_accuracy, overall_f1, per_class_f1 = self.evaluate()
            val_overall_accuracies.append(overall_accuracy)
            val_overall_f1.append(overall_f1)
            val_f1_per_class.append(per_class_f1)
            val_losses.append(val_loss)

            self.model.train()

            # === Scheduler step ===
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)  # based on validation loss
                else:
                    self.scheduler.step()

    def predict(self):
        """
        Make predictions using the trained model.
        For:
        - SimpleCNN: returns a list of predicted class indices.
        - MobileNetV3: returns a list of detections per image, each with 'boxes' and 'labels'.
        """

        self.model.eval()
        predictions = []

        with torch.no_grad():
            for batch in tqdm(self.test_loader):
                if self.model_name == "SimpleCNN":
                    
                    images, parent_ids, patch_names = batch
            
                    images = images.to(self.device)
                    outputs = self.model(images)
                    preds = torch.argmax(outputs, dim=1).cpu().tolist()

                    for parent, patch, pred in zip(parent_ids, patch_names, preds):
                        predictions.append({
                            "image_id": parent,
                            "patch_name": patch,
                            "pred_class": pred
                        })

                elif self.model_name == "MobileNetV3":
                    # batch: list of image tensors
                    if isinstance(batch, (tuple, list)):
                        images = batch[0]  # Drop labels if included
                    else:
                        images = batch

                    # Convert to list if batch is a single tensor
                    if isinstance(images, torch.Tensor):
                        images = [images[i] for i in range(images.size(0))]

                    image_tensors = [img.to(self.device) for img in images]
                    outputs = self.model(image_tensors)

                    for out in outputs:
                        boxes = out["boxes"].cpu()
                        scores = out["scores"].cpu()
                        labels = out["labels"].cpu()

                        keep = scores > 0.6
                        predictions.append({
                            'boxes': boxes[keep],
                            'labels': labels[keep],
                            'scores': scores[keep]
                        })

                elif self.model_name == "SSDLiteMobileNetV3":
                    images, filenames = batch
                    images = list(img.to(self.device) for img in images)
                    outputs = self.model(images)

                    for filename, output in zip(filenames, outputs):
                        boxes = output["boxes"].cpu()
                        scores = output["scores"].cpu()
                        labels = output["labels"].cpu()

                        keep = scores > 0.1
                        predictions.append({
                            "filename": filename,
                            "boxes": boxes[keep],
                            "labels": labels[keep],
                            "scores": scores[keep]
                        })

                else:
                    raise ValueError(f"Unknown model name: {self.model_name}")

        return predictions

    def save_model(self, DIR="output/", model_name=None):
        """
        Save the model to the specified directory.
        """
        if model_name is None:
            model_name = self.model_name

        if not os.path.exists(DIR):
            os.makedirs(DIR)
            
        torch.save(self.model.state_dict(), os.path.join(DIR, f'{model_name}.pth'))

    def load_model(self, model_name):
        """
        Load the model from the specified directory.
        """
        if os.path.exists(os.path.join(DIR, model_name)):
            self.model.load_state_dict(torch.load(os.path.join(DIR, model_name)))
        else:
            raise FileNotFoundError("Model file not found.")
        
    def compute_iou(self, boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

        iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
        return iou

    def evaluate(self):
        """
        Evaluate the model on the validation set.
        Computes:
        - Overall accuracy
        - Overall F1 score (weighted)
        - Per-class F1 scores
        Supports:
        - SimpleCNN (classification)
        - MobileNetV3 (object detection: uses predicted/target class labels)
        """
        self.model.eval()

        all_preds = 0
        all_preds_list = []
        all_targets = []
        total_loss = 0.0
        num_batches = 0
        total_gts = 0
        correct = 0


        with torch.no_grad():
            for images, labels in tqdm(self.val_loader):

                if self.model_name == "SimpleCNN":
                    # Classification
                    inputs = images.to(self.device)
                    targets = labels.to(self.device)

                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)
                    preds = torch.argmax(outputs, dim=1)

                    all_preds_list.extend(preds.cpu().tolist())
                    all_targets.extend(targets.cpu().tolist())

                # elif self.model_name == "MobileNetV3":
                #     # Object detection
                #     image_tensors = [img.to(self.device) for img in images]
                #     targets = [{k: v.to(self.device) for k, v in t.items()} for t in labels]

                #     loss_dict = self.model(image_tensors, targets)
                #     loss = sum(loss for loss in loss_dict.values())

                #     outputs = self.model(image_tensors)  # for predictions

                #     for pred, target in zip(outputs, targets):
                #         pred_labels = pred['labels'].detach().cpu().tolist()
                #         true_labels = target['labels'].detach().cpu().tolist()

                #         all_preds.extend(pred_labels)
                #         all_targets.extend(true_labels)

                elif self.model_name == "SSDLiteMobileNetV3" or self.model_name == "MobileNetV3":
                    iou_threshold = 0.5
                    targets = labels

                    image_tensors = [img.to(self.device) for img in images]
                    targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

                    # === Get loss for validation ===
                    self.model.train()
                    loss_dict = self.model(image_tensors, targets)
                    loss = sum(loss for loss in loss_dict.values())
                    total_loss += loss.item()
                    num_batches += 1
                    self.model.eval()

                    # === Get predictions ===
                    outputs = self.model(image_tensors)

                    for pred, target in zip(outputs, targets):
                        pred_boxes = pred['boxes'].cpu()
                        pred_scores = pred['scores'].cpu()
                        pred_labels = pred['labels'].cpu()
                        gt_boxes = target['boxes'].cpu()
                        gt_labels = target['labels'].cpu()

                        # Filter low-confidence predictions (optional)
                        keep = pred_scores > 0.6
                        pred_boxes = pred_boxes[keep]
                        pred_labels = pred_labels[keep]

                        all_preds += len(pred_labels)
                        total_gts += len(gt_labels)

                        matched_pred_idxs = set()
                        for gt_idx, gt_box in enumerate(gt_boxes):
                            best_iou = 0
                            best_pred_idx = -1
                            for pred_idx, pred_box in enumerate(pred_boxes):
                                if pred_idx in matched_pred_idxs:
                                    continue
                                iou = self.compute_iou(gt_box, pred_box)
                                if iou > best_iou:
                                    best_iou = iou
                                    best_pred_idx = pred_idx
                            if best_iou >= iou_threshold:
                                matched_pred_idxs.add(best_pred_idx)
                                if pred_labels[best_pred_idx] == gt_labels[gt_idx]:
                                    correct += 1

                
                else:
                    raise ValueError(f"Unknown model name: {self.model_name}")

            total_loss += loss.item()
            print(f"Batch [{num_batches}], Loss: {total_loss:.4f}")

            num_batches += 1


        if self.model_name == "SSDLiteMobileNetV3" or self.model_name == "MobileNetV3":
            avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
            precision = correct / all_preds if all_preds > 0 else 0.0
            recall = correct / total_gts if total_gts > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall + 1e-6)
            return avg_loss, precision, f1, None

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        overall_accuracy = accuracy_score(all_targets, all_preds_list)
        overall_f1 = f1_score(all_targets, all_preds_list, average='weighted')
        per_class_f1 = f1_score(all_targets, all_preds_list, average=None)

        return avg_loss, overall_accuracy, overall_f1, per_class_f1
        
        