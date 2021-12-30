import torch.nn as nn
import pytorch_lightning as pl
import torchmetrics.functional
import torch.optim as optim
from utils.utils import inverse_norm
from config import config


class NormalGRU(pl.LightningModule):
    def __init__(self):
        super(NormalGRU, self).__init__()
        self.loss_module = nn.MSELoss()

        self.GRU = nn.GRU(input_size=config.num_features,
                          hidden_size=config.normal_hidden_size,
                          num_layers=config.normal_num_layers)
        self.Linear = nn.Linear(in_features=config.normal_hidden_size, out_features=1)

    def forward(self, x):
        out, h = self.GRU(x)
        out = self.Linear(out)
        out = out.squeeze(-1)
        return out

    def training_step(self, batch, batch_idx):
        X, y_true = batch
        y_pred = self.forward(X)

        loss = self.loss_module(y_pred, y_true)
        return loss

    def validation_step(self, batch, batch_idx):
        X, y_true = batch
        y_pred = self(X)
        y_true, y_pred = inverse_norm(y_true), inverse_norm(y_pred)
        mape = torchmetrics.functional.mean_absolute_percentage_error(y_pred, y_true)
        self.log(name='val_mape', value=mape)

    def test_step(self, batch, batch_idx):
        X, y_true = batch
        y_pred = self(X)
        y_true, y_pred = inverse_norm(y_true), inverse_norm(y_pred)
        mape = torchmetrics.functional.mean_absolute_percentage_error(y_pred, y_true)
        self.log(name='test_mape', value=mape)

    def configure_optimizers(self):
        opt = optim.Adam(self.parameters(), lr=config.normal_lr)
        return opt






