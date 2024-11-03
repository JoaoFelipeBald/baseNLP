import matplotlib.pyplot as plt
import numpy as np
import torchvision
import torch
from sklearn.metrics import f1_score
import random
from torch_lr_finder import LRFinder
from torch.amp import autocast, GradScaler



# Initialize the scaler only if a GPU is available
scaler = GradScaler() if torch.cuda.is_available() else None

def set_seed(seed=42):
    '''
    Set seeds for experiment consistency and reproducibility
    '''
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    random.seed(42)
set_seed()

import os
import torch
import wandb
from tqdm.auto import tqdm
from typing import Dict, List, Tuple
import torch.optim.lr_scheduler as lr_scheduler
import random
class train:
    # Constructor method (called when an object is created)
    def __init__(self, model, config,
                 loss, device,
                 train_loader, test_loader,
                 checkpoints, project_name, step_factor,
                 optimizer, max_norm=1):
        self.model=model
        self.config=config
        self.project_name=project_name
        self.loss_fn=loss
        self.device=device
        self.train_loader=train_loader
        self.test_loader=test_loader
        self.checkpoints=checkpoints
        self.step_factor=step_factor
        self.optimizer=optimizer
        self.max_norm=max_norm
        self.x=1
        # After training
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, 'checkpoint.pth')

    from sklearn.metrics import f1_score
    from typing import Tuple
    import torch
    from torch.cuda.amp import autocast
    from sklearn.metrics import f1_score

    def train_step(self, model: torch.nn.Module,
              dataloader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module,
              optimizer: torch.optim.Optimizer,
              device: torch.device,
              scaler=None) -> Tuple[float, float, float]:
        """Trains a PyTorch model for a single epoch and returns train_loss, train_accuracy, train_f1."""

        model.train()
        train_loss, correct_predictions, total_samples = 0, 0, 0
        all_preds, all_labels = [], []

        for batch, (input_ids, attention_mask, y) in enumerate(dataloader):
            # Move inputs and labels to the device (CPU or GPU)
            input_ids, attention_mask, y = input_ids.to(device), attention_mask.to(device), y.to(device)
            if scaler is not None:  # Mixed precision training
                with autocast(device_type='cuda'):
                    # Forward pass: pass both input_ids and attention_mask to the model
                    y_pred = model(input_ids=input_ids, attention_mask=attention_mask)
                    loss = loss_fn(y_pred, y)

                train_loss += loss.item()
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.max_norm)  # Gradient clipping
                scaler.step(optimizer)
                scaler.update()
            else:  # Standard training
                y_pred = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = loss_fn(y_pred, y)

                train_loss += loss.item()
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.max_norm)
                optimizer.step()

            # Predictions and accuracy calculation
            y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
            correct_predictions += (y_pred_class == y).sum().item()
            total_samples += y.size(0)

            # Collect all predictions and labels for F1-score calculation
            all_preds.extend(y_pred_class.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

        # Calculate metrics
        train_loss /= len(dataloader)
        train_acc = correct_predictions / total_samples
        train_f1 = f1_score(all_labels, all_preds, average='weighted')

        return train_loss, train_acc, train_f1

    from typing import Tuple
    import torch
    from sklearn.metrics import f1_score

    def test_step(self, model: torch.nn.Module,
                  dataloader: torch.utils.data.DataLoader,
                  loss_fn: torch.nn.Module,
                  device: torch.device) -> Tuple[float, float, float]:
        """Tests a PyTorch model for a single epoch and returns test_loss, test_accuracy, test_f1."""

        model.eval()
        test_loss, correct_predictions, total_samples = 0, 0, 0
        all_preds, all_labels = [], []

        with torch.inference_mode():
            for batch, (input_ids, attention_mask, y) in enumerate(dataloader):
                # Move inputs and labels to the device (CPU or GPU)
                input_ids, attention_mask, y = input_ids.to(device), attention_mask.to(device), y.to(device)

                # Forward pass: pass both input_ids and attention_mask to the model
                test_pred_logits = model(input_ids=input_ids, attention_mask=attention_mask)

                # Loss calculation
                loss = loss_fn(test_pred_logits, y)
                test_loss += loss.item()

                # Predictions and accuracy calculation
                test_pred_labels = torch.argmax(test_pred_logits, dim=1)
                correct_predictions += (test_pred_labels == y).sum().item()
                total_samples += y.size(0)

                # Collect all predictions and labels for F1-score calculation
                all_preds.extend(test_pred_labels.cpu().numpy())
                all_labels.extend(y.cpu().numpy())

        # Calculate metrics
        test_loss /= len(dataloader)
        test_acc = correct_predictions / total_samples
        test_f1 = f1_score(all_labels, all_preds, average='weighted')

        return test_loss, test_acc, test_f1




    class EarlyStopping:
        def __init__(self, patience: int, verbose: bool = False):
            """
            Args:
                patience (int): Number of epochs to wait for improvement before stopping.
                verbose (bool): If True, prints a message when early stopping is triggered.
            """
            self.patience = patience
            self.verbose = verbose
            self.best_loss = float('inf')
            self.epochs_no_improve = 0
            self.early_stop = False

        def __call__(self, current_loss: float):
            """
            Call this method at the end of each epoch to check if early stopping should be triggered.

            Args:
                current_loss (float): The current test loss to compare with the best loss.
            """
            if current_loss < self.best_loss:
                self.best_loss = current_loss
                self.epochs_no_improve = 0
            else:
                self.epochs_no_improve += 1
            print("Times without improvements: "+str(self.epochs_no_improve))
            if self.epochs_no_improve >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print(f"Early stopping triggered after {self.patience} epochs without improvement.")

    def get_current_lr(self, optimizer):
        """Helper function to get the current learning rate from the optimizer."""
        for param_group in optimizer.param_groups:
            return param_group['lr']

    def save_checkpoint(self, state, filename="model_checkpoint.pth"):
        """Saves the model and optimizer state."""
        print(f"=> Saving checkpoint to {filename}")
        torch.save(state, filename)

    def train2(self, model: torch.nn.Module,
            train_dataloader: torch.utils.data.DataLoader,
            test_dataloader: torch.utils.data.DataLoader,
            optimizer: torch.optim.Optimizer,
            loss_fn: torch.nn.Module,
            epochs: int,
            device: torch.device,
            patience: int,
            patience_step: int,
            factor: int,
            checkpoint_path: str = "checkpoints") -> Dict[str, List]:
        """Trains and tests a PyTorch model with early stopping, learning rate scheduler, and checkpoints.

        Args:
        model: A PyTorch model to be trained and tested.
        train_dataloader: A DataLoader instance for the model to be trained on.
        test_dataloader: A DataLoader instance for the model to be tested on.
        optimizer: A PyTorch optimizer to help minimize the loss function.
        loss_fn: A PyTorch loss function to calculate loss on both datasets.
        epochs: An integer indicating how many epochs to train for.
        device: A target device to compute on (e.g. "cuda" or "cpu").
        patience: Number of epochs to wait for improvement before stopping early.
        checkpoint_path: Path to save the best model checkpoint.

        Returns:
        A dictionary of training and testing loss as well as training and testing accuracy metrics.
        """

        id=random.random()
        # Create empty results dictionary
        results = {"train_loss": [], "train_acc": [], "train_f1": [],
               "test_loss": [], "test_acc": [], "test_f1": []}

        # Initialize early stopping and learning rate scheduler
        early_stopping = self.EarlyStopping(patience=patience, verbose=True)
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=factor, patience=patience_step, verbose=True)

        wandb.init(project=self.project_name, name=f"run-name{self.x}", config={
            "epochs": epochs,
            "batch_size": train_dataloader.batch_size,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "architecture": model.__class__.__name__,
        })

        # Ensure model is on target device
        model.to(device)

        # Initialize best loss for checkpointing
        best_loss = float('inf')

        # Loop through training and testing steps for a number of epochs
        for epoch in tqdm(range(epochs)):
            # Training step
            train_loss, train_acc, train_f1 = self.train_step(model=model,
                                                          dataloader=train_dataloader,
                                                          loss_fn=loss_fn,
                                                          optimizer=optimizer,
                                                          device=device)

            # Testing step
            test_loss, test_acc, test_f1 = self.test_step(model=model,
                                                      dataloader=test_dataloader,
                                                      loss_fn=loss_fn,
                                                      device=device)

            wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "train_f1": train_f1,
            "test_loss": test_loss,
            "test_acc": test_acc,
            "test_f1": test_f1
            })

            # Print out what's happening
            print(f"Epoch: {epoch+1} | train_loss: {train_loss:.4f} | train_acc: {train_acc:.4f} | "
              f"train_f1: {train_f1:.4f} | test_loss: {test_loss:.4f} | test_acc: {test_acc:.4f} | "
              f"test_f1: {test_f1:.4f}")

            # Save model checkpoint if test loss improves
            if test_loss < best_loss:
                path=os.path.join(checkpoint_path, f"best_model_run_{self.x}.pth")
                print(f"Test loss improved from {best_loss:.4f} to {test_loss:.4f}. Saving checkpoint...")
                best_loss = test_loss
                self.save_checkpoint({
                    'epoch': epoch + 1,  # Save current epoch
                    'model_state_dict': model.state_dict(),  # Save model parameters
                    'optimizer_state_dict': optimizer.state_dict(),  # Save optimizer state
                    'loss': test_loss,  # Save the loss
                    'scheduler': scheduler.state_dict()  # Save scheduler state
                }, filename=path)

            # Call early stopping
            early_stopping(test_loss)

            if early_stopping.early_stop:
                print("Early stopping triggered")
                break
            current_lr = self.get_current_lr(optimizer)
            print(f"Before scheduler.step() - Epoch: {epoch+1}, Learning Rate: {current_lr}")
            # Step the scheduler with the test loss
            scheduler.step(test_loss)
            updated_lr = self.get_current_lr(optimizer)
            print(f"After scheduler.step() - Epoch: {epoch+1}, Updated Learning Rate: {updated_lr}")
            # Update results dictionary
            results["train_loss"].append(train_loss)
            results["train_acc"].append(train_acc)
            results["test_loss"].append(test_loss)
            results["test_acc"].append(test_acc)
            wandb.watch(model)
            #torch.save(model.state_dict(), f"model_state/model_epoch_{epoch+1}.pth")
            wandb.save(f"model_epoch_{epoch+1}.pth")  # Save checkpoint to WandB

        self.x+=1
        wandb.finish()
        # Return the filled results at the end of the epochs
        return results
    # Define a wrapper function that initializes everything and then calls `train2`
    def sweep_train(self):
        # Initialize W&B for this run
        wandb.init()


        config = wandb.config
        optimizer = self.optimizer
        checkpoint = torch.load('checkpoint.pth')
        self.model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        optimizer.param_groups[0]["lr"] = config.learning_rate

        # Call train2 with the appropriate arguments
        results=self.train2(model=self.model,
            train_dataloader=self.train_loader,
            test_dataloader=self.test_loader,
            optimizer=optimizer,
            loss_fn=self.loss_fn,
            epochs=config.epochs,
            device=config.device,
            patience=config.patience,
            patience_step=config.patience_step,
            factor=self.step_factor)
        return results

    # Initialize the sweep
    def train_model(self, count):
        sweep_id = wandb.sweep(sweep=self.config, project=self.project_name)
        # Start the timer
        from timeit import default_timer as timer
        start_time = timer()
        print(self.device)
        # Run the sweep agent
        wandb.agent(sweep_id, function=self.sweep_train, count=count)
        # End the timer and print out how long it took
        end_time = timer()

def find_lr(optimizer, model, train_dataloader, loss_fn, device, end_lr, num_iter):
      # Create a learning rate finder instance
      lr_finder = LRFinder(model, optimizer, loss_fn, device=device)

      # Perform the learning rate range test
      lr_finder.range_test(train_dataloader, end_lr=end_lr, num_iter=num_iter)

      # Plot the learning rate against the loss to find the optimal learning rate
      lr_finder.plot()
