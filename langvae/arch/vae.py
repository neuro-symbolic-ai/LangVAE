import torch
import torch.nn.functional as F
from pythae.models.vae import VAE


class LangVAE(VAE):
    def loss_function(self, recon_x, x, mu, log_var, z):
        x = torch.squeeze(x)
        x_tok_ids = torch.argmax(x, dim=-1)
        mask = (x_tok_ids != self.decoder.tokenizer.pad_token_id).to(torch.int8)

        if self.model_config.reconstruction_loss == "mse":
            recon_loss = (0.5 * F.mse_loss(recon_x, x, reduction="none").sum(dim=-1) * mask).sum(dim=-1)

        elif self.model_config.reconstruction_loss == "bce":
            recon_loss = (F.nll_loss(torch.log(recon_x).view(recon_x.shape[0] * recon_x.shape[1], recon_x.shape[2]),
                                     x_tok_ids.view(recon_x.shape[0] * recon_x.shape[1]),
                                     reduction="none").sum(dim=-1) * mask).sum(dim=-1)

        KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=-1)

        return (recon_loss + KLD).mean(dim=0), recon_loss.mean(dim=0), KLD.mean(dim=0)