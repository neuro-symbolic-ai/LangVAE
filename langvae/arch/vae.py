import numpy as np
import torch
import torch.nn.functional as F
from typing import Tuple, List, Optional

from pythae.trainers import BaseTrainerConfig
from pythae.models.nn import BaseEncoder, BaseDecoder
from torch import Tensor
from pythae.models.vae import VAE, VAEConfig
from pythae.trainers.training_callbacks import TrainingCallback


@torch.jit.script
def vae_nll_loss(recon_x: Tensor,
                 x: Tensor,
                 mu: Tensor,
                 log_var: Tensor,
                 z: Tensor,
                 pad_token_id: int,
                 beta: float,
                 target_kl: float) -> Tuple[Tensor, Tensor, Tensor]:
    """
        Calculates the negative log-likelihood (NLL) loss for a Variational Autoencoder (VAE).

        Args:
            recon_x (Tensor): The reconstructed input tensor.
            x (Tensor): The original input tensor.
            mu (Tensor): The mean of the latent variable distribution.
            log_var (Tensor): The logarithm of the variance of the latent variable distribution.
            z (Tensor): The latent variable tensor.
            pad_token_id (int): The padding token ID for the input sequence.
            beta (float): A hyperparameter that controls the trade-off between reconstruction loss and KL divergence.
            target_kl (float): A target value for the KL divergence (cut-off).

        Returns:
            Tuple[Tensor, Tensor, Tensor]:
                - Total NLL loss (reconstruction loss + KL divergence).
                - Average reconstruction loss.
                - Average KL divergence.
        """
    x = torch.squeeze(x).to(recon_x.device)
    x_tok_ids = torch.argmax(x, dim=-1)
    mask = (x_tok_ids != pad_token_id).to(torch.int8)

    recon_loss = (F.nll_loss(torch.log(recon_x).view(recon_x.shape[0] * recon_x.shape[1], recon_x.shape[2]),
                             x_tok_ids.view(recon_x.shape[0] * recon_x.shape[1]),
                             reduction="none").sum(dim=-1) * mask).sum(dim=-1) / x.shape[0]

    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=-1)
    kl_mask = (KLD > target_kl).float()
    KLD = beta * (kl_mask * KLD)

    return (recon_loss + KLD).mean(dim=0), recon_loss.mean(dim=0), KLD.mean(dim=0)

@torch.jit.script
def vae_nll_loss_supervised(recon_x: Tensor,
                            x: Tensor,
                            mu: Tensor,
                            log_var: Tensor,
                            z: Tensor,
                            pad_token_id: int,
                            beta: float,
                            target_kl: float,
                            num_annotations: int) -> Tuple[Tensor, Tensor, Tensor]:
    x = torch.squeeze(x).to(recon_x.device)
    x_split = x.chunk(num_annotations + 1, dim=-1)
    recon_x_split = recon_x.chunk(num_annotations + 1, dim=-1)
    x_tok_ids = torch.argmax(x_split[0], dim=-1)
    mask = (x_tok_ids != pad_token_id).to(torch.int8)
    rec_x = recon_x_split[0]

    recon_loss = (F.nll_loss(torch.log(rec_x).view(rec_x.shape[0] * rec_x.shape[1], rec_x.shape[2]),
                             x_tok_ids.view(rec_x.shape[0] * rec_x.shape[1]),
                             reduction="none").sum(dim=-1) * mask).sum(dim=-1) / x_split[0].shape[0]

    for lbl_split, rec_lbl in zip(x_split[1:], recon_x_split[1:]):
        x_lbl_ids = torch.argmax(lbl_split, dim=-1)
        mask = (x_lbl_ids != pad_token_id).to(torch.int8)
        recon_loss += (F.nll_loss(torch.log(rec_lbl).view(rec_lbl.shape[0] * rec_lbl.shape[1], rec_lbl.shape[2]),
                                  x_lbl_ids.view(rec_lbl.shape[0] * rec_lbl.shape[1]),
                                  reduction="none").sum(dim=-1) * mask).sum(dim=-1) / lbl_split.shape[0]

    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=-1)
    kl_mask = (KLD > target_kl).float()
    KLD = beta * (kl_mask * KLD)


    return (recon_loss + KLD).mean(dim=0), recon_loss.mean(dim=0), KLD.mean(dim=0)


class LangVAE(VAE):
    """
    A language-oriented Variational Autoencoder (VAE) that can be used for text generation.

    Args:
        model_config (VAEConfig): The configuration of the VAE model.
        encoder (Optional[BaseEncoder]): Language encoder model to be used.
        decoder (Optional[BaseDecoder]): Language decoder model to be used.
    """

    def __init__(
            self,
            model_config: VAEConfig,
            encoder: Optional[BaseEncoder],
            decoder: Optional[BaseDecoder]
    ):
        super().__init__(model_config=model_config, encoder=encoder, decoder=decoder)
        self.cur_beta: float = 0.0
        self.target_kl = 1.0

    def loss_function(self, recon_x, x, mu, log_var, z) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Computes the loss function for the VAE model.

        Args:
            recon_x (Tensor): The reconstructed input tensor.
            x (Tensor): The original input tensor.
            mu (Tensor): The mean of the latent variable distribution.
            log_var (Tensor): The logarithm of the variance of the latent variable distribution.
            z (Tensor): The sampled latent variable tensor.

        Returns:
            Tuple[Tensor, Tensor, Tensor]: A tuple containing the reconstruction loss, the KL divergence
                loss, and the total loss.
        """
        if (recon_x.shape[-1] > self.decoder.decoder.config.vocab_size):
            num_annotations = recon_x.shape[-1] // self.decoder.decoder.config.vocab_size - 1
            losses = vae_nll_loss_supervised(recon_x, x, mu, log_var, z, self.decoder.tokenizer.pad_token_id,
                                             self.cur_beta, self.target_kl, num_annotations)
        else:
            losses = vae_nll_loss(recon_x, x[:,:,:self.decoder.decoder.config.vocab_size], mu, log_var, z,
                                  self.decoder.tokenizer.pad_token_id, self.cur_beta, self.target_kl)
        print("\n", [l.item() for l in losses])
        return losses

    def encode_z(self, x: Tensor) -> Tensor:
        """
        Encodes the input tensor into a latent variable tensor.

        Args:
            x (Tensor): The input tensor to be encoded.

        Returns:
            Tensor: A tensor containing the sampled latent variables.
        """
        encoded = self.encoder(x)
        mu, log_var = encoded["embedding"], encoded["log_covariance"]
        std = torch.exp(0.5 * log_var)
        z, eps = self._sample_gauss(mu, std)

        return z

    def decode_sentences(self, z: Tensor) -> List[str]:
        """
        Decodes the latent variable tensor into a list of sentences.

        Args:
            z (Tensor): The latent variable tensor to be decoded.

        Returns:
            List[str]: A list of strings representing the decoded sentences.
        """
        generated = self.decoder(z)["reconstruction"]
        sents = self.decoder.tokenizer.batch_decode(torch.argmax(generated, dim=-1), skip_special_tokens=True)

        return sents

    def push_to_hf_hub(self, hf_hub_path: str):
        """
        Uploads the VAE model to the Hugging Face Hub.

        Args:
            hf_hub_path (str): The HF hub path where the model should be uploaded to.
        """
        self.device = "cpu"
        self.encoder.device = "cpu"
        self.decoder.device = "cpu"
        self.encoder.debug = False
        self.decoder.debug = False
        super().push_to_hf_hub(hf_hub_path)
