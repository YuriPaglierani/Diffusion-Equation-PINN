import torch
import warnings
from pathlib import Path
from torch.nn import Module
from typing import Optional

def load_model(model: Module, batch_size: int=64, lr_gamma: float=0.965, lr: float=1e-3, lr_step: int=1, add_name: str=''):
    """
    Load the most recent checkpoint for the given model.

    Args:
        model (Module): The model to load the checkpoint into.
        batch_size (int): The batch size used in training. Defaults to 64.
        lr_gamma (float): Learning rate decay factor. Defaults to 0.965.
        lr (float): Learning rate. Defaults to 1e-3.
        lr_step (int): Learning rate step size in epochs. Defaults to 1.
        add_name (str): Additional string to append to the model's name for directory naming. Defaults to an empty string.

    Returns:
        Optional[Module]: The model with loaded state dict if checkpoint is found, otherwise None.
    """

    def get_last_ckpt_dir(model: Module) -> Optional[Path]:
        """
        Get the directory of the last checkpoint for the given model.

        Args:
            model (Module): The model to find the checkpoint for.

        Returns:
            Optional[Path]: Path to the last checkpoint directory or None if no checkpoints are found.
        """

        def get_model_dir(model: Module) -> Path:
            """
            Get the directory where the model's checkpoints are stored.

            Args:
                model (Module): The model to find the directory for.

            Returns:
                Path: Path to the model's checkpoint directory.
            """
            
            return Path(
                    "result",
                    f"{model.name}{add_name}-bs{batch_size}-lr{lr}-lrstep{lr_step}"
                    f"-lrgamma{lr_gamma}",
                )
        
        output_dir = get_model_dir(model)
        ckpts = list(output_dir.glob("ckpt-*"))
        if len(ckpts) == 0:
            return None
        
        ckpts = sorted(ckpts, key=lambda x: int(x.stem.split('-')[-1]))
        return ckpts[-1]

    last_ckpt_dir = get_last_ckpt_dir(model)
    if last_ckpt_dir is not None:
        print(f"Resuming from {last_ckpt_dir}")
        model.load_state_dict(torch.load(last_ckpt_dir / "ckpt.pt"))
        return model
    warnings.warn("No model checkpoint found, returned None", UserWarning)
    return None
