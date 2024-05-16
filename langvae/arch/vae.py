import sys
import os
import pickle
import numpy as np
import torch
import torch.nn.functional as F
from typing import Tuple, List, Optional
from copy import deepcopy

from pythae.trainers import BaseTrainerConfig
from pythae.models.nn import BaseEncoder, BaseDecoder
from pythae.models.base.base_config import BaseAEConfig, EnvironmentConfig
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
        encoder (Optional[BaseEncoder]): Language encoder model that processes input data and returns sentence embeddings.
        decoder (Optional[BaseDecoder]): Language decoder model that generates text from latent representations.
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

    def save(self, dir_path: str):
        """Method to save the model at a specific location. It saves, the model weights as a
        ``models.pt`` file along with the model config as a ``model_config.json`` file. If the
        model to save used custom encoder (resp. decoder) provided by the user, these are also
        saved as ``decoder.pkl`` (resp. ``decoder.pkl``).

        Args:
            dir_path (str): The path where the model should be saved. If the path
                path does not exist a folder will be created at the provided location.
        """

        env_spec = EnvironmentConfig(
            python_version=f"{sys.version_info[0]}.{sys.version_info[1]}"
        )
        model_dict = {"model_state_dict": deepcopy(self.state_dict())}

        if not os.path.exists(dir_path):
            try:
                os.makedirs(dir_path)

            except FileNotFoundError as e:
                raise e

        env_spec.save_json(dir_path, "environment")
        self.model_config.save_json(dir_path, "model_config")

        # only save .pkl if custom architecture provided
        if not self.model_config.uses_default_encoder:
            with open(os.path.join(dir_path, "encoder.pkl"), "wb") as fp:
                # cloudpickle.register_pickle_by_value(inspect.getmodule(self.encoder))
                pickle.dump(self.encoder, fp, pickle.DEFAULT_PROTOCOL)

        if not self.model_config.uses_default_decoder:
            with open(os.path.join(dir_path, "decoder.pkl"), "wb") as fp:
                # cloudpickle.register_pickle_by_value(inspect.getmodule(self.decoder))
                pickle.dump(self.decoder, fp, pickle.DEFAULT_PROTOCOL)

        torch.save(model_dict, os.path.join(dir_path, "model.pt"))
