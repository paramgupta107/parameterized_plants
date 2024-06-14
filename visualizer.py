import json
import torch
import os

class Visualizer:
    """
    Visualizer class for saving timelapse data.

    Args:
        logs_path (str, optional): The path to save the optimization logs. Defaults to 'logs'.
    
    """
    def __init__(self, logs_path: str = 'log.json'):
        self.logs_path = logs_path
        self.timelapse = {"steps":[], "epoch":[], "loss":[]}
    def add_target(self, targetParams: torch.Tensor):
        """
        Add the target parameters to the timelapse. Can only add once.

        Args:
            targetParams (torch.Tensor): The target parameters of shape (1, 17).

        Returns:
            None
        """
        if "target" in self.timelapse:
            raise ValueError("Target already exists.")
        self.timelapse["target"] = targetParams[0].tolist()
    
    def add_step(self, stepParams: torch.Tensor, epoch: int, loss: float = 0):
        """
        Add a step to the timelapse.
        
        Args:
            stepParams (torch.Tensor): The step parameters of shape (1, 17).
            epoch (int): The epoch of the step.
            loss (float, optional): The loss of the step. Defaults to 0.
        """
        self.timelapse["steps"].append(stepParams[0].tolist())
        self.timelapse["epoch"].append(epoch)
        self.timelapse["loss"].append(loss)
    def save(self):
        """
        Save the timelapse data to a JSON file. Must have added a target and at least one step.

        Returns:
            None
        """
        if len(self.timelapse["steps"]) == 0 or "target" not in self.timelapse:
            raise ValueError("No data to save.")
        directory = os.path.dirname(self.logs_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
        with open(self.logs_path, 'w') as file:
            json.dump(self.timelapse, file)
