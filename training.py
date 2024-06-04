import json
import torch
from torch import nn

from torch.utils.data import DataLoader
from model import Pinn
from pathlib import Path
from time import time
from typing import Dict, Any, Optional

def dump_json(path: Path, data: Dict[str, Any]) -> None:
    """
    Dump data to a JSON file.

    Args:
        path (Path): The file path where the JSON file will be saved.
        data (Dict[str, Any]): The data to be saved in JSON format.
    """

    with open(path, "w") as f:
        json.dump(data, f, indent=4)

class Standardize(nn.Module):
    """
    A module to standardize inputs based on provided mean and standard deviation.
    """

    def __init__(self, mean: torch.Tensor, std: torch.Tensor):
        """
        Initialize the Standardize module.

        Args:
            mean (torch.Tensor): The mean values for standardization.
            std (torch.Tensor): The standard deviation values for standardization.
        """

        super(Standardize, self).__init__()
        self.mean = mean
        self.std = std
        final_std = torch.sqrt(std[0]*std[1])
        self.std[0] = self.std[1] = final_std
        self.conversion = final_std**2/self.std[2]
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to standardize the input tensor.

        Args:
            x (torch.Tensor): Input tensor to be standardized.

        Returns:
            torch.Tensor: Standardized tensor.
        """

        return (x - self.mean) / self.std

class Trainer:
    """Trainer for convenient training and testing of a PINN model."""
    
    def __init__(self, model: Pinn, mean_train: torch.Tensor, std_train: torch.Tensor,
                 num_epochs: int = 5, batch_size: int = 64*4, log_interval: int = 100):
        """
        Initialize the Trainer.

        Args:
            model (Pinn): The PINN model to be trained.
            mean_train (torch.Tensor): The mean values for training data standardization.
            std_train (torch.Tensor): The standard deviation values for training data standardization.
            num_epochs (int): Number of training epochs. Defaults to 5.
            batch_size (int): Batch size for training. Defaults to 256.
            log_interval (int): Interval for logging training progress. Defaults to 100.
        """
        
        self.model = model

        # Hyperparameters
        self.lr = 1e-3
        self.lr_step = 1  # Unit is epoch
        self.lr_gamma = 0.965
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.log_interval = log_interval
        self.step_size = 200
        self.standardize = Standardize(mean_train, std_train)
        self.input_keys = ['x', 'y', 't', 'u']
        self.train_full_loss_history = []
        self.train_data_loss_history = []
        self.train_pde_loss_history = []
        self.val_full_loss_history = []
        self.val_data_loss_history = []
        self.val_pde_loss_history = []
        self.diffusion_history = []
        self.output_dir = Path(
            "result",
            f"{model.name}-bs{self.batch_size}-lr{self.lr}-lrstep{self.lr_step}"
            f"-lrgamma{self.lr_gamma}",
        )

        print(f"Output dir: {self.output_dir}")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        args = {}
        for attr in ["lr", "lr_step", "lr_gamma", "num_epochs", "batch_size"]:
            args[attr] = getattr(self, attr)
        dump_json(self.output_dir / "args.json", args)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3, weight_decay=1e-4)
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=self.step_size, gamma=self.lr_gamma
        )


    def get_last_ckpt_dir(self) -> Optional[Path]:
        """
        Get the directory of the last checkpoint.

        Returns:
            Optional[Path]: The path to the last checkpoint directory, or None if no checkpoints exist.
        """
        
        ckpts = list(self.output_dir.glob("ckpt-*"))
        if len(ckpts) == 0:
            return None
        ckpts = sorted(ckpts, key=lambda x: int(x.stem.split('-')[-1]))
        return ckpts[-1]
    
    def init_setting(self, last_ckpt_dir: Optional[Path]) -> int:
        """
        Initialize model settings from the last checkpoint if available.

        Args:
            last_ckpt_dir (Optional[Path]): The directory of the last checkpoint.

        Returns:
            int: The starting epoch.
        """

        if last_ckpt_dir is not None:
            print(f"Resuming from {last_ckpt_dir}")
            self.model.load_state_dict(torch.load(last_ckpt_dir / "ckpt.pt"))
            self.optimizer.load_state_dict(
                torch.load(last_ckpt_dir / "optimizer.pt")
            )
            self.lr_scheduler.load_state_dict(
                torch.load(last_ckpt_dir / "lr_scheduler.pt")
            )
        
            return int(last_ckpt_dir.name.split("-")[-1]) + 1
        
        return 0

    def set_inputs(self, batch: torch.Tensor, device: torch.device) -> Dict[str, torch.Tensor]:
        """
        Standardize input batch and set requires_grad to True for each tensor.

        Args:
            batch (torch.Tensor): The input batch tensor.
            device (torch.device): The device to transfer the batch to.

        Returns:
            Dict[str, torch.Tensor]: A dictionary of input tensors with requires_grad set to True.
        """

        std_batch = self.standardize(batch[0].to(device))
        return {self.input_keys[i]: std_batch[:, i].requires_grad_(True) for i in range(std_batch.shape[-1])}

    def display_stat(self, step: int, loss: torch.Tensor, val_full_loss: float, train_start_time: float) -> None:
        """
        Display training statistics.

        Args:
            step (int): The current training step.
            loss (torch.Tensor): The current training loss.
            val_full_loss (float): The current validation loss.
            train_start_time (float): The starting time of the training.
        """

        print(
            {
                "step": step,
                "loss": round(loss.item(), 6),
                "val_loss": round(val_full_loss, 6),
                "lr": round(
                    self.lr_scheduler.get_last_lr()[0], 6
                ),
                # "diffusion_coeff_inside": round(self.model.diffusion_coeff.item(), 4),
                "diffusion_coeff_true": round(self.standardize.conversion.item() * self.model.diffusion_coeff.item(), 4),
                "time": round(time() - train_start_time, 1),
            }
        )
    
    def dumper(self) -> None:
        """
        Dump training and validation loss histories to JSON files.
        """

        dump_json(self.output_dir / "train_loss_history.json", 
                  {"full": self.train_full_loss_history,
                  "data": self.train_data_loss_history,
                  "pde": self.train_pde_loss_history})
        dump_json(self.output_dir / "val_loss_history.json", 
                  {"full": self.val_full_loss_history,
                  "data": self.val_data_loss_history,
                  "pde": self.val_pde_loss_history})
        dump_json(self.output_dir / "diffusion_history.json", 
                  {"diffusion": self.diffusion_history})

    def train(self, train_loader: DataLoader, validation_loader: Optional[DataLoader] = None) -> None:
        """
        Train the model with the given data loaders.

        Args:
            train_loader (DataLoader): The data loader for training data.
            validation_loader (Optional[DataLoader]): The data loader for validation data.
        """

        device = self.device

        print("====== Training ======")
        print(f"# epochs: {self.num_epochs}")
        print(f"# examples: {len(train_loader)*self.batch_size}")
        print(f"batch size: {self.batch_size}")
        print(f"# steps: {len(train_loader)}")
        
        self.model.train()
        self.model.to(device)

        # Resume
        last_ckpt_dir = self.get_last_ckpt_dir()
        ep = self.init_setting(last_ckpt_dir)

        train_start_time = time()
        while ep < self.num_epochs:
            print(f"====== Epoch {ep} ======")
            for step, batch in enumerate(train_loader):
    
                inputs = self.set_inputs(batch, device)
    
                # Forward
                _, loss, losses = self.model(**inputs).values()
            
                self.train_full_loss_history.append(loss.item())
                self.train_data_loss_history.append(losses['u_loss'].item())
                self.train_pde_loss_history.append(losses['f_u_loss'].item())

                # Backward
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

                if step % self.log_interval == 0:
                    val_full_loss = 0
                    val_data_loss = 0
                    val_pde_loss = 0
                    if validation_loader:
                        for val_batch in validation_loader:
                            val_inputs = self.set_inputs(val_batch, device)
                            _, val_loss, val_losses = self.model(**val_inputs).values()
                            
                            val_full_loss += val_loss.item()
                            val_data_loss += val_losses["u_loss"].item()
                            val_pde_loss += val_losses["f_u_loss"].item() 

                        val_full_loss /= len(validation_loader)
                        val_data_loss /= len(validation_loader)
                        val_pde_loss /= len(validation_loader)
                        self.val_full_loss_history.append(val_full_loss)
                        self.val_data_loss_history.append(val_data_loss)
                        self.val_pde_loss_history.append(val_pde_loss)
                    
                    self.display_stat(step, loss, val_full_loss, train_start_time)
     
                self.diffusion_history.append(self.standardize.conversion.item() * round(self.model.diffusion_coeff.item(), 4))
                self.lr_scheduler.step()
            ep = self.checkpoint(ep)
            print(f"====== Epoch {ep} done ======")
        print("====== Training done ======")
        # self.dumper()

    def checkpoint(self, ep: int) -> int:
        """
        Dump checkpoint (model, optimizer, lr_scheduler) to "ckpt-{ep}" in
        the `output_dir`,

        and dump `self.loss_history` to "loss_history.json" in the
        `ckpt_dir`, and clear `self.loss_history`.
        """
        # Evaluate and save
        ckpt_dir = self.output_dir / f"ckpt-{ep}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        print(f"Checkpointing to {ckpt_dir}")
        torch.save(self.model.state_dict(), ckpt_dir / "ckpt.pt")
        torch.save(self.optimizer.state_dict(), ckpt_dir / "optimizer.pt")
        torch.save(
            self.lr_scheduler.state_dict(), ckpt_dir / "lr_scheduler.pt"
        )
                  
        return ep+1