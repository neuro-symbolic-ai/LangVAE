import sys
import os
import json
import logging
import numpy as np
import pythae.models.base.base_utils
import torch
import torch.nn.functional as F
from typing import Tuple, List, Dict, Optional
from copy import deepcopy
from dataclasses import asdict

from pythae.models.base.base_config import EnvironmentConfig
from pythae.models.base.base_utils import ModelOutput
from pythae.data.datasets import BaseDataset
from torch import Tensor
from pythae.models.wae_mmd import WAE_MMD, WAE_MMD_Config
from pythae.models.base.base_utils import hf_hub_is_available

from langvae.encoders import SentenceEncoder
from langvae.decoders import SentenceDecoder
from langvae.data_conversion.sparse import densify_w_padding
from nltk.translate.bleu_score import sentence_bleu

logger = logging.getLogger(__name__)
console = logging.StreamHandler()
logger.addHandler(console)
logger.setLevel(logging.INFO)

model_card_template = """---
language: en
tags:
- langvae
license: apache-2.0
---

### Downloading this model from the Hub
This model was trained with {clsname}. It can be downloaded or reloaded using the method `load_from_hf_hub`
```python
>>> from langvae import {clsname}
>>> model = {clsname}.load_from_hf_hub(hf_hub_path="your_hf_username/repo_name")
```
"""


def calc_bleu(decoded: dict) -> float:
    bleu_scores = list()

    for sent in decoded:
        bleu_scores.append(sentence_bleu([ref.split() for ref in decoded[sent]], sent.split()))

    bleu_scores = [score for score in bleu_scores if (score > 0)]

    return sum(bleu_scores) / (len(bleu_scores) + 1e-8)


# def imq_kernel(X: torch.Tensor,
#                Y: torch.Tensor,
#                h_dim: int):
#     batch_size = X.shape[0]
#     device = X.device
#
#     norms_x = X.pow(2).sum(1, keepdim=True)  # batch_size x 1
#     prods_x = torch.mm(X, X.t())  # batch_size x batch_size
#     dists_x = norms_x + norms_x.t() - 2 * prods_x
#
#     norms_y = Y.pow(2).sum(1, keepdim=True)  # batch_size x 1
#     prods_y = torch.mm(Y, Y.t())  # batch_size x batch_size
#     dists_y = norms_y + norms_y.t() - 2 * prods_y
#
#     dot_prd = torch.mm(X, Y.t())
#     dists_c = norms_x + norms_y.t() - 2 * dot_prd
#
#     stats = 0
#     for scale in [.1, .2, .5, 1., 2., 5., 10.]:
#         C = 2 * h_dim * 1.0 * scale
#         res1 = C / (C + dists_x)
#         res1 += C / (C + dists_y)
#         res1 = (1 - torch.eye(batch_size, device=device)) * res1
#         res1 = res1.sum() / (batch_size - 1)
#         res2 = C / (C + dists_c)
#         res2 = res2.sum() * 2. / (batch_size)
#         stats += res1 - res2
#
#     return stats

# def imq_kernel(z1: Tensor, z2: Tensor, latent_dim: int):
#     """Returns a matrix of shape [batch x batch] containing the pairwise kernel computation"""
#
#     Cbase = (
#         2.0 * latent_dim
#     )
#
#     k = 0
#
#     for scale in [0.1, 0.2, 0.5, 1.0, 2.0, 5, 10.0]:
#         C = scale * Cbase
#         k += C / (C + torch.norm(z1.unsqueeze(1) - z2.unsqueeze(0), dim=-1) ** 2)
#
#     return k


@torch.compile
def vae_nll_loss(recon_x: Tensor, x: Tensor, pad_token_id: int) -> Tensor:
    """
    Calculates the negative log-likelihood (NLL) loss for an LM Autoencoder (AE).

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

    if (x.layout == torch.sparse_coo):
        x_tok_ids = densify_w_padding(x, pad_token_id)
        # x_tok_ids = [x[i].coalesce().indices()[1][:len] for i in range(x.shape[0])]
        # x_tok_ids = torch.stack([
        #     torch.cat([tok_ids, torch.tensor([pad_token_id] * int(tok_ids.shape[0] < len) +
        #                                      [0] * max(len - tok_ids.shape[0] - 1, 0),
        #                                      dtype=torch.int64, device=x.device)
        #     ])
        #     for tok_ids in x_tok_ids
        # ])
        mask = (x_tok_ids != pad_token_id).to(torch.int8)
    else:
        x_tok_ids = x.argmax(dim=-1)
        mask = (x_tok_ids != pad_token_id).to(torch.int8)

    mask = torch.logical_or(mask, mask.roll(1, dims=1)).to(torch.int8)

    recon_loss = (F.nll_loss(torch.log(recon_x).view(recon_x.shape[0] * recon_x.shape[1], recon_x.shape[2]),
                             x_tok_ids.view(recon_x.shape[0] * recon_x.shape[1]),
                             reduction="none").view(x.shape[:2]) * mask).sum(dim=-1)

    # print(f"recon x: {recon_x.mean(dim=0)}")
    # print(f"recon loss: {recon_loss.mean(dim=0)}, ")

    print("X:", x_tok_ids[:2, :10])
    print("R:", recon_x.argmax(dim=-1)[:2, :10])

    return recon_loss.mean(dim=0)


class LangWAE(WAE_MMD):
    """
    A language-oriented Wasserstein Autoencoder (WAE) that can be used for text generation.

    Args:
        model_config (WAE_MMD_Config): The configuration of the VAE model.
        encoder (Optional[SentenceEncoder]): Language encoder model that processes input data and returns sentence embeddings.
        decoder (Optional[SentenceDecoder]): Language decoder model that generates text from latent representations.
    """

    # loss_writer = SummaryWriter()

    def __init__(
            self,
            model_config: WAE_MMD_Config,
            encoder: Optional[SentenceEncoder],
            decoder: Optional[SentenceDecoder]
    ):
        super().__init__(model_config=model_config, encoder=encoder, decoder=decoder)

        # Logging losses
        self.debug = False
        self._dbg_counter = 0
        self._loss_agg = [0.0, 0.0]

    @staticmethod
    def _sample_gauss(mu, std):
        # Reparametrization trick
        # Sample N(0, I)
        eps = torch.randn_like(std)
        return mu + eps * std, eps

    def forward(self, inputs: BaseDataset, **kwargs):
        """
        The VAE model

        Args:
            inputs (BaseDataset): The training dataset with labels

        Returns:
            ModelOutput: An instance of ModelOutput containing all the relevant parameters

        """

        x = inputs["data"]
        x_annot = inputs.keys() - {"data", "input_ids", "attention_mask"}
        x_teach = x if (self.decoder.teacher_forcing and self.decoder.training) else None
        cvars = None
        if (x_annot):
            cvars = {annot: inputs[annot] for annot in x_annot}

        encoded = self.encoder(x, cvars)

        mu, log_var = encoded.embedding, encoded.log_covariance
        cvars_emb = encoded.cvars_embedding

        std = torch.exp(0.5 * log_var)
        z, eps = LangWAE._sample_gauss(mu, std)
        z = torch.cat([z] + cvars_emb, dim=-1) if (cvars and self.decoder.conditional) else z

        recon_x = self.decoder(z, max_len=x.shape[1], x=x_teach)["reconstruction"]

        loss, recon_loss, mmd_loss = self.loss_function(recon_x, x, z)

        output = ModelOutput(
            recon_loss=recon_loss,
            mmd_loss=mmd_loss,
            loss=loss,
            recon_x=recon_x,
            z=z,
        )

        return output

    @torch.compile
    def mmd_loss(self, z: Tensor) -> Tensor:
        N = z.shape[0]  # batch size
        z_prior = torch.randn_like(z)

        if self.kernel_choice == "rbf":
            k_z = self.rbf_kernel(z, z)
            k_z_prior = self.rbf_kernel(z_prior, z_prior)
            k_cross = self.rbf_kernel(z, z_prior)

        else:
            k_z = self.imq_kernel(z, z)
            k_z_prior = self.imq_kernel(z_prior, z_prior)
            k_cross = self.imq_kernel(z, z_prior)

        mmd_z = (k_z - k_z.diag().diag()).sum() / ((N - 1) * N)
        mmd_z_prior = (k_z_prior - k_z_prior.diag().diag()).sum() / ((N - 1) * N)
        mmd_cross = k_cross.sum() / (N ** 2)

        mmd_loss = mmd_z + mmd_z_prior - 2 * mmd_cross

        return mmd_loss

    def loss_function(self, recon_x, x, z) -> Tuple[Tensor, Tensor, Tensor]:
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
        recon_x.clamp_min_(torch.finfo(recon_x.dtype).tiny * 10)  # Prevents underflow

        recon_loss = vae_nll_loss(recon_x, x, self.decoder.tokenizer.pad_token_id)
        mmd_loss = self.mmd_loss(z)
        print("MMD Loss:", mmd_loss)


        generated = recon_x.argmax(dim=-1)
        for i in range(generated.shape[0]):
            eos_idx = (generated[i] == self.decoder.tokenizer.pad_token_id).nonzero()
            if (len(eos_idx) > 0):
                generated[i, eos_idx[0]:] = self.decoder.tokenizer.pad_token_id

        x_tok_ids = densify_w_padding(x, self.decoder.tokenizer.pad_token_id)
        orig_sents = self.decoder.tokenizer.batch_decode(x_tok_ids, skip_special_tokens=True)
        recon_sents = self.decoder.tokenizer.batch_decode(generated, skip_special_tokens=True)
        decoded = {os: [rs] for os, rs in zip(orig_sents, recon_sents)}
        print("BLEU:", calc_bleu(decoded))

        return recon_loss + mmd_loss * self.model_config.reg_weight, recon_loss, mmd_loss

    def encode_z(self, x: Tensor, c: Dict[str, Tensor] = None) -> Tuple[Tensor, List[Tensor]]:
        """
        Encodes the input tensor into a latent variable tensor.

        Args:
            x (Tensor): The input tensor to be encoded.

        Returns:
            Tuple[Tensor, List[Tensor]]: A tuple of tensors containing the sampled latent variables and conditional
            variable embeddings if available, respectively.
        """
        encoded = self.encoder(x, c)
        mu, log_var = encoded.embedding, encoded.log_covariance
        cvars_emb = encoded.cvars_embedding
        std = torch.exp(0.5 * log_var)
        z, eps = self._sample_gauss(mu, std)

        return (z, cvars_emb)

    def decode_sentences(self, z: Tensor, cvars_emb: List[Tensor] = None) -> List[str]:
        """
        Decodes the latent variable tensor into a list of sentences.

        Args:
            z (Tensor): The latent variable tensor to be decoded.

        Returns:
            List[str]: A list of strings representing the decoded sentences.
        """
        z = torch.cat([z] + cvars_emb, dim=-1) if (cvars_emb and self.decoder.conditional) else z
        generated = self.decoder(z)["reconstruction"]

        # Propagates the first padding token to the rest of the sequence.
        for i in range(generated.shape[0]):
            eos_idx = (generated[i] == self.decoder.tokenizer.pad_token_id).nonzero()
            if (len(eos_idx) > 0):
                generated[i, eos_idx[0]:] = self.decoder.tokenizer.pad_token_id

        sents = self.decoder.tokenizer.batch_decode(generated.argmax(dim=-1), skip_special_tokens=True)

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
        self.debug = False
        self.encoder.debug = False
        self.decoder.debug = False
        pythae.models.base.base_utils.model_card_template = model_card_template.format(clsname=self.__class__.__name__)
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

        if not os.path.exists(dir_path):
            try:
                os.makedirs(dir_path)

            except FileNotFoundError as e:
                raise e

        env_spec.save_json(dir_path, "environment")
        self.model_config.save_json(dir_path, "model_config")

        if not self.model_config.uses_default_encoder:
            torch.save(self.encoder.state_dict(), os.path.join(dir_path, "encoder.pt"))
            with open(os.path.join(dir_path, "encoder_cfg.json"), "w") as enc_cfg_file:
                json.dump({"model_path": self.encoder.model_path,
                           "latent_size": self.encoder.latent_size,
                           "automodel_preset": asdict(self.encoder.automodel_preset),
                           "caching": self.encoder.caching}, enc_cfg_file)

        if not self.model_config.uses_default_decoder:
            torch.save(self.decoder.state_dict(), os.path.join(dir_path, "decoder.pt"))
            with open(os.path.join(dir_path, "decoder_cfg.json"), "w") as dec_cfg_file:
                cfg = {
                    "model_path": self.decoder.model_path,
                    "latent_size": self.decoder.latent_size,
                    "max_len": self.decoder.max_len,
                    "conditional": self.decoder.conditional,
                    "memory_factor": self.decoder.memory_factor,
                    "teacher_forcing": self.decoder.teacher_forcing,
                    "device_map": self.decoder.device_map
                }
                json.dump(cfg, dec_cfg_file)

    @classmethod
    def _load_custom_encoder_from_folder(cls, dir_path, tokenizer):

        file_list = os.listdir(dir_path)
        cls._check_python_version_from_folder(dir_path=dir_path)

        if "encoder_cfg.json" not in file_list:
            raise FileNotFoundError(
                f"Missing encoder config file ('encoder_cfg.json') in"
                f"{dir_path}... This file is needed to rebuild custom encoders."
                " Cannot perform model building."
            )

        else:
            with open(os.path.join(dir_path, "encoder_cfg.json"), "r") as fp:
                cfg = json.load(fp)
            with open(os.path.join(dir_path, "encoder.pt"), "rb") as fp:
                encoder = SentenceEncoder(**(cfg | {"decoder_tokenizer": tokenizer}))
                encoder.load_state_dict(torch.load(fp, map_location=torch.device(encoder.device), weights_only=True))

            if (not encoder._encoder):
                encoder.init_pretrained_model()


        return encoder

    @classmethod
    def _load_custom_decoder_from_folder(cls, dir_path):

        file_list = os.listdir(dir_path)
        cls._check_python_version_from_folder(dir_path=dir_path)

        if "decoder_cfg.json" not in file_list:
            raise FileNotFoundError(
                f"Missing decoder config file ('decoder_cfg.json') in"
                f"{dir_path}... This file is needed to rebuild custom decoders."
                " Cannot perform model building."
            )

        else:
            with open(os.path.join(dir_path, "decoder_cfg.json"), "r") as fp:
                cfg = json.load(fp)
            with open(os.path.join(dir_path, "decoder.pt"), "rb") as fp:
                decoder = SentenceDecoder(**cfg)
                decoder.load_state_dict(torch.load(fp, map_location=torch.device(decoder.device), weights_only=True))

        return decoder

    @classmethod
    def load_from_folder(cls, dir_path):
        """Class method to be used to load the model from a specific folder

        Args:
            dir_path (str): The path where the model should have been be saved.

        .. note::
            This function requires the folder to contain:

            - | a ``model_config.json`` and a ``model.pt`` if no custom architectures were provided

            **or**

            - | a ``model_config.json``, a ``model.pt`` and a ``encoder.pkl`` (resp.
                ``decoder.pkl``) if a custom encoder (resp. decoder) was provided
        """

        model_config = cls._load_model_config_from_folder(dir_path)
        decoder = cls._load_custom_decoder_from_folder(dir_path)
        encoder = cls._load_custom_encoder_from_folder(dir_path, decoder.tokenizer)

        model = cls(model_config, encoder=encoder, decoder=decoder)

        return model

    @classmethod
    def load_from_hf_hub(cls, hf_hub_path: str):  # pragma: no cover
        """Class method to be used to load a pretrained model from the Hugging Face hub

        Args:
            hf_hub_path (str): The path where the model should have been be saved on the
                hugginface hub.

        .. note::
            This function requires the folder to contain:

            - | a ``model_config.json`` and a ``model.pt`` if no custom architectures were provided

            **or**

            - | a ``model_config.json``, a ``model.pt`` and a ``encoder.pkl`` (resp.
                ``decoder.pkl``) if a custom encoder (resp. decoder) was provided
        """

        if not hf_hub_is_available():
            raise ModuleNotFoundError(
                "`huggingface_hub` package must be installed to load models from the HF hub. "
                "Run `python -m pip install huggingface_hub` and log in to your account with "
                "`huggingface-cli login`."
            )

        else:
            from huggingface_hub import hf_hub_download

        logger.info(f"Downloading {cls.__name__} files for rebuilding...")

        _ = hf_hub_download(repo_id=hf_hub_path, filename="environment.json")
        config_path = hf_hub_download(repo_id=hf_hub_path, filename="model_config.json")
        dir_path = os.path.dirname(config_path)

        model_config = cls._load_model_config_from_folder(dir_path)

        if not model_config.uses_default_decoder:
            _ = hf_hub_download(repo_id=hf_hub_path, filename="decoder_cfg.json")
            _ = hf_hub_download(repo_id=hf_hub_path, filename="decoder.pt")
            decoder = cls._load_custom_decoder_from_folder(dir_path)
        else:
            decoder = None

        if not model_config.uses_default_encoder:
            _ = hf_hub_download(repo_id=hf_hub_path, filename="encoder_cfg.json")
            _ = hf_hub_download(repo_id=hf_hub_path, filename="encoder.pt")
            encoder = cls._load_custom_encoder_from_folder(dir_path, decoder.tokenizer)
        else:
            encoder = None


        logger.info(f"Successfully downloaded {cls.__name__} model!")

        model = cls(model_config, encoder=encoder, decoder=decoder)

        return model