import os
import json
import torch
import numpy as np
from dawo import DAWO, loss_function, Anndata_to_Tensor


class DAWOWrapper:
    """
    Minimal wrapper for DAWO model to use with Hugging Face Hub
    """
    def __init__(self, repo_path):
        """
        Initialize the DAWO model
        
        Args:
            repo_path: Path to repository with model files
        """
        # Load configuration
        config_path = os.path.join(repo_path, "config.json")
        with open(config_path, 'r') as f:
            config = json.load(f)
            
        # Create model with original DAWO class
        self.model = DAWO(
            input_dim_X=config["input_dim_X"],
            input_dim_Y=config["input_dim_Y"],
            input_dim_Z=config["input_dim_Z"],
            latent_dim=config["latent_dim"],
            Y_emb=config["Y_emb"],
            Z_emb=config["Z_emb"],
            num_classes=config["num_classes"]
        )
        
        # Load weights
        self.model.load_state_dict(torch.load(os.path.join(repo_path, "model.pth")))
        self.model.eval()
    
    def predict(self, x, y, z):
        """
        Make predictions with the DAWO model
        
        Args:
            x: Gene expression tensor (batch_size, input_dim_X)
            y: Drug feature tensor (batch_size, input_dim_Y)
            z: Cell line feature tensor (batch_size, input_dim_Z)
            
        Returns:
            Dict with model outputs
        """
        with torch.no_grad():
            x_hat, mu, logvar, y_pred = self.model(x, y, z)
            
        return {
            "x_hat": x_hat,              # Reconstructed gene expression
            "mu": mu,                    # Latent mean
            "logvar": logvar,            # Latent log variance
            "y_pred": y_pred,            # Drug response predictions
            "probs": torch.softmax(y_pred, dim=1)  # Drug response probabilities
        } 