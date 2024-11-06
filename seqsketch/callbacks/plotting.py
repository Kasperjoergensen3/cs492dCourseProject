from pytorch_lightning.callbacks import Callback
from torch.utils.data import DataLoader
import wandb
import matplotlib.pyplot as plt
import numpy as np
import torch


class ImageLogger(Callback):
    def __init__(self, model, data_module):
        super().__init__()

        dataloader = data_module.val_dataloader()
        batch = next(iter(dataloader))
        self.x, self.c = model.prepare_batch(batch)

    def plot_inline(self, condition_image, sample_image, true_image):
        fig, ax = plt.subplots(1, 3, figsize=(10, 5))
        fig.tight_layout()
        for im, title, a in zip(
            [condition_image, sample_image, true_image],
            ["Condition", "Sample", "True"],
            ax,
        ):
            if im is None:
                im = np.zeros(self.x.shape[2:])
            a.imshow(im.squeeze().detach().cpu().numpy(), cmap="gray", vmin=0, vmax=1)
            a.set_title(title)
            a.axis("off")
        # convert to wandb image
        wandb_im = wandb.Image(fig)
        plt.close()
        return wandb_im

    def on_validation_epoch_end(self, trainer, pl_module):
        # Dataloader loads on CPU --> pass to GPU
        x0 = self.x.to(device=pl_module.device)
        c = self.c
        if c is not None:
            c = self.c.to(device=pl_module.device)
        zt = torch.randn_like(x0)
        x0_pred = pl_module.sampling(zt, c)

        # move arrays back to the CPU for plotting
        x0_pred = x0_pred.cpu()
        x0 = x0.cpu()
        if c is not None:
            c = c.cpu()

        # generate figures in a list
        figs = [
            self.plot_inline(im1, im2, im3) for im1, im2, im3 in zip(c, x0_pred, x0)
        ]

        # add to logger like so
        trainer.logger.experiment.log({"Sample images": figs})
