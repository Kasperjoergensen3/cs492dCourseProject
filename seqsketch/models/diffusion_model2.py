import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from seqsketch.utils.modules import get_class_from_string
from tqdm import tqdm


class SeqStrokeDiffusionModule2(pl.LightningModule):

    def __init__(
        self,
        params,
    ):
        super().__init__()
        self.params = params
        self._initialize_networks()
        self._intialize_scheduler()
        self.num_train_timesteps = params.scheduler.params.num_train_timesteps
        self.num_inference_timesteps = params.num_inference_timesteps
        self.input_channels = params.denoising_network.params.input_channels
        self.image_resolution = params.denoising_network.params.image_resolution

        self.threshold = 0.5

    def load_pretrained_weights(self, path):
        # Load the checkpoint data (this includes more than just the model's state dict)
        checkpoint = torch.load(path, map_location=self.device)

        # Extract and load only the model's state dictionary
        self.load_state_dict(checkpoint["state_dict"])

    def _intialize_scheduler(self):
        cls = get_class_from_string(f"seqsketch.models.{self.params.scheduler.module}")
        self.scheduler = cls(self.params.scheduler.params)

    def _initialize_networks(self):
        cls = get_class_from_string(
            f"seqsketch.models.{self.params.denoising_network.module}"
        )
        self.denoising_network = cls(self.params.denoising_network.params)
        self.learnable_parameters = self.denoising_network.parameters()

    def calculate_loss(self, x0, c):
        t = torch.randint(
            0, self.num_train_timesteps, (x0.size(0),), device=x0.device
        ).long()
        eps = torch.randn_like(x0)
        xt = self.scheduler.add_noise(x0, eps, t)
        eps_pred = self.denoising_network(xt, t, c)
        loss = torch.mean((eps_pred - eps) ** 2)
        print(loss)
        return loss

    def sampling(self, c):
        # run full denoising process
        x0_pred = torch.randn_like(c[0])
        for t in tqdm(self.scheduler.timesteps):
            t = t.repeat(x0_pred.size(0))
            eps_pred = self.denoising_network(x0_pred, t, c)
            x0_pred = self.scheduler.step(eps_pred, t[0], x0_pred).prev_sample
        return self.postprocess(x0_pred)

    def postprocess(self, x_hat):
        # clamp tp 0,256 and round to nearest integer
        x_hat = torch.clamp(x_hat, 0, 1)
        return x_hat

    def forward(self, bs, iterations=10):
        c = [torch.zeros((bs, 32, 2), device=self.device) for _ in range(32)]
        for i in range(iterations):
            x_hat = self.sampling(c)
            c[i] = x_hat
        return c[:iterations]

    def prepare_batch(self, batch):
        # here we should prepare batch for the model
        x = batch["next_stroke"]  # shape: (batch_size, channels, height, width)
        x_mask = batch["next_stroke_mask"]
        # c = batch["current_strokes"]  # shape: (batch_size, channels, height, width)
        # n_strokes = batch["n_strokes"]
        # step = batch["step"]
        if "current_strokes" in self.params.conditioning:
            c = batch["current_strokes"]
            c_mask = batch["current_strokes_mask"]
        else:
            c = None
            c_mask = None
        return x, x_mask, c, c_mask

    def training_step(self, batch, batch_idx):
        x, c = self.prepare_batch(batch)
        loss = self.calculate_loss(x, c)
        # log the loss
        self.log("train_loss", loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, c = self.prepare_batch(val_batch)
        loss = self.calculate_loss(x, c)
        # log the loss
        self.log("val_loss", loss)
        pass

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.learnable_parameters, lr=self.params.lr)
        if self.params.lr_scheduler:
            if self.params.lr_scheduler == "exponential_decay_0.01":
                optimizer = torch.optim.lr_scheduler.ExponentialLR(
                    optimizer, gamma=0.01
                )
            else:
                raise NotImplementedError(
                    f"Learning rate scheduler {self.params.lr_scheduler} not implemented"
                )
        return optimizer

    def train(self, mode=True):
        super().train(mode)
        if mode:
            self.scheduler.set_timesteps(self.num_train_timesteps)
            self.scheduler.timesteps = self.scheduler.timesteps.to(self.device)

    def eval(self):
        super().eval()
        self.scheduler.set_timesteps(self.num_inference_timesteps)
        self.scheduler.timesteps = self.scheduler.timesteps.to(self.device)
