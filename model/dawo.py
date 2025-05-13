import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, TensorDataset
import torch.nn.functional as F


def Anndata_to_Tensor(adata, label=None, label_continuous= None ,batch=None, device='cpu'):
    # sparse matrix to tensor
    if isinstance(adata.X, (sp.csr.csr_matrix, sp.csc.csc_matrix)):
        X_tensor = torch.tensor(adata.X.toarray(), dtype=torch.float32).to(device)
    else:
        X_tensor = torch.tensor(adata.X, dtype=torch.float32).to(device)

    tensors = {'X_tensor': X_tensor}

    if label is not None:
        labels_num, _ = pd.factorize(adata.obs[label], sort=True)
        tensors['labels_num'] = torch.tensor(labels_num, dtype=torch.long)
    
    if label_continuous is not None:
        tensors['label_continuous'] = torch.tensor(adata.obs[label_continuous], dtype=torch.float64)

    if batch is not None:
        batch_one_hot = pd.get_dummies(adata.obs[batch]).to_numpy()
        tensors['batch_one_hot'] = torch.from_numpy(batch_one_hot)

    if len(tensors) == 1 and 'X_tensor' in tensors:
        return tensors['X_tensor']
    else:
        # return TensorDataset with available tensors
        return TensorDataset(*tensors.values())


def loss_function(x_hat, x, mu, logvar, β=0.1):
    BCE = nn.functional.mse_loss(
        x_hat, x.view(-1, x_hat.shape[1]), reduction='sum'
    )
    KLD = 0.5 * torch.sum(logvar.exp() - logvar - 1 + mu.pow(2))

    return BCE+  β * KLD

class DAWO(nn.Module):
    def __init__(self, input_dim_X, input_dim_Y, input_dim_Z, latent_dim_mid=500, latent_dim=50, Y_emb=50, Z_emb=50, num_classes=10):
        super(DAWO, self).__init__()

        self.encoder = nn.Sequential(
            nn.BatchNorm1d(input_dim_X),
            nn.Linear(input_dim_X, latent_dim_mid),
            nn.ReLU(),
            nn.Dropout(0.2),  
            nn.Linear(latent_dim_mid, latent_dim * 2),
        )

        self.encoder_Y = nn.Sequential(
            nn.BatchNorm1d(input_dim_Y),
            nn.Linear(input_dim_Y, latent_dim_mid),
            nn.ReLU(),
            nn.Dropout(0.2),  
            nn.Linear(latent_dim_mid, Y_emb),
        )

        self.encoder_Z = nn.Sequential(
            nn.BatchNorm1d(input_dim_Z),
            nn.Linear(input_dim_Z, latent_dim_mid),
            nn.ReLU(),
            nn.Dropout(0.2),   
            nn.Linear(latent_dim_mid, Z_emb),
        )



        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + Y_emb + Z_emb, latent_dim_mid),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(latent_dim_mid, input_dim_X),
        )

        self.classifier = nn.Sequential(
            nn.BatchNorm1d(latent_dim + Z_emb),
            nn.Linear(latent_dim + Z_emb, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

        self.input_dim = input_dim_X
        self.input_dim_Y = input_dim_Y
        self.input_dim_Z = input_dim_Z
        self.latent_dim = latent_dim

    def reparameterise(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, x, y, z):
        mu_logvar = self.encoder(x.view(-1, self.input_dim)).view(-1, 2, self.latent_dim)
        l_y = self.encoder_Y(y.view(-1, self.input_dim_Y)) 
        l_z = self.encoder_Z(z.view(-1, self.input_dim_Z)) 

        mu = mu_logvar[:, 0, :]
        logvar = mu_logvar[:, 1, :]
        l_x = self.reparameterise(mu, logvar)

        l_xyz = torch.cat((l_x, l_y, l_z), dim=1)
        l_xz = torch.cat((l_x, l_z), dim=1)

        x_hat = self.decoder(l_xyz)
        y_pred = self.classifier(l_xz)

        return x_hat, mu, logvar, y_pred
