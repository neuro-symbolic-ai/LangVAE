import torch
from typing import Tuple, List, Optional, Sequence

from pythae.models.base.base_model import ModelOutput
from pythae.data.datasets import BaseDataset
from torch import nn, Tensor
from pythae.models.vae import VAEConfig
from langvae.arch.vae import LangVAE
from langvae.encoders import AnnotatedSentenceEncoder
from langvae.decoders import SentenceDecoder


class LangCVAE(LangVAE):
    """
    A subclass of LangVAE implementing a Conditional Variational Autoencoder (CVAE).
    This model extends LangVAE by incorporating conditional variables into the latent space, allowing for
    controlled text generation based on specified input annotations.

    Attributes:
        emb2z (nn.Linear): A linear transformation layer used to project encoded annotations into the latent space.

    Args:
        model_config (VAEConfig): Configuration settings for the VAE model.
        encoder (Optional[AnnotatedSentenceEncoder]): Language encoder model that processes input data and returns sentence embeddings.
        decoder (Optional[SentenceDecoder]): The decoder that generates text from latent representations.
    """
    def __init__(
            self,
            model_config: VAEConfig,
            encoder: Optional[AnnotatedSentenceEncoder],
            decoder: Optional[SentenceDecoder]
    ):
        decoder = type(decoder)(decoder.decoder.config.name_or_path,
                                encoder.encoder.config.hidden_size * (encoder.num_annotations + 1),
                                decoder.max_len,
                                decoder.device,
                                decoder.load_in_4bit,
                                decoder.device_map,
                                decoder.max_look_behind)
        decoder.debug = True
        super().__init__(model_config=model_config, encoder=encoder, decoder=decoder)
        self.z2emb = nn.Linear(model_config.latent_dim, encoder.encoder.config.hidden_size, device=self.device)

    def forward(self, inputs: BaseDataset, **kwargs):
        """
        Defines the forward pass of the LangCVAE model.

        Args:
            inputs (BaseDataset): The input dataset entries containing text data and conditional variables.

        Returns:
            ModelOutput: A structured output containing loss values, reconstructed data, and latent variables.
        """
        x = inputs["data"]

        encoder_output = self.encoder(x)
        mu, log_var = encoder_output.embedding, encoder_output.log_covariance
        std = torch.exp(0.5 * log_var)
        z, eps = self._sample_gauss(mu.to(self.device), std.to(self.device))
        cond_vars = encoder_output["embedding_lbls"]
        recon_x = self.decoder(torch.cat([torch.cat((self.z2emb(z), cvar.to(self.device)), dim=-1) for cvar in cond_vars], dim=-1))["reconstruction"]

        loss, recon_loss, kld = self.loss_function(recon_x, x, mu, log_var, z)

        output = ModelOutput(
            recon_loss=recon_loss,
            reg_loss=kld,
            loss=loss,
            recon_x=recon_x,
            z=z,
        )

        return output

    def encode_z(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Encodes input data into latent variables and conditional variables.

        Args:
            x (Tensor): Input tensor to be encoded.

        Returns:
            Tuple[Tensor, Tensor]: A tuple containing the latent variables and conditional variables.
        """
        encoded = self.encoder(x)
        mu, log_var = encoded["embedding"], encoded["log_covariance"]
        cond_vars = encoded["embedding_lbls"]
        std = torch.exp(0.5 * log_var)
        z, eps = self._sample_gauss(mu, std)

        return z, cond_vars

    def decode_sentences(self, z: Tensor, cond_vars: Sequence[Tensor]) -> List[str]:
        """
        Decodes latent and conditional variables into sentences.

        Args:
            z (Tensor): Latent variables.
            cond_vars (Sequence[Tensor]): Conditional variables associated with the latent variables.

        Returns:
            List[str]: A list of generated sentences.
        """
        cond_z = torch.cat([torch.cat((self.z2emb(z), cvar), dim=-1) for cvar in cond_vars], dim=-1)
        generated = self.decoder(cond_z)["reconstruction"]
        sents = self.decoder.tokenizer.batch_decode(torch.argmax(generated, dim=-1), skip_special_tokens=True)

        return sents

