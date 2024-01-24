from pythae.pipelines import TrainingPipeline
from pythae.models import VAEConfig
from pythae.trainers import BaseTrainerConfig
from saf_datasets import CPAEDataSet
from langvae.arch.vae import LangVAE
from langvae.encoders import SentenceEncoder
from langvae.decoders import SentenceDecoder
from langvae.data_conversion.tokenization import TokenizedDataSet

DEVICE = "cuda:1"
LATENT_SIZE = 32
MAX_SENT_LEN = 20


def main():
    cpae = CPAEDataSet()
    decoder = SentenceDecoder("gpt2", LATENT_SIZE, MAX_SENT_LEN, device=DEVICE)
    # decoder = SentenceDecoder("mistralai/Mixtral-8x7B-v0.1", LATENT_SIZE, MAX_SENT_LEN, device=DEVICE,
    #                           load_in_4bit=True, device_map="balanced_low_0")
    encoder = SentenceEncoder("bert-base-cased", LATENT_SIZE, decoder.tokenizer, device=DEVICE)
    train_dataset = TokenizedDataSet(cpae[:10000], decoder.tokenizer, decoder.max_len, device=DEVICE)
    eval_dataset = TokenizedDataSet(cpae[10000:10100], decoder.tokenizer, decoder.max_len, device=DEVICE)

    model_config = VAEConfig(
        input_dim=(train_dataset[0]["data"].shape[-2], train_dataset[0]["data"].shape[-1]),
        latent_dim=LATENT_SIZE
    )
    model = LangVAE(model_config, encoder, decoder)

    training_config = BaseTrainerConfig(
        output_dir='my_vae',
        num_epochs=200,
        learning_rate=1e-4,
        per_device_train_batch_size=40,
        per_device_eval_batch_size=40,
        steps_saving=None,
        optimizer_cls="Adam",
        # optimizer_params={"weight_decay": 0.05, "betas": (0.91, 0.995)},
        scheduler_cls="ReduceLROnPlateau",
        scheduler_params={"patience": 5, "factor": 0.5}
    )

    pipeline = TrainingPipeline(
        training_config=training_config,
        model=model
    )

    pipeline(
        train_data=train_dataset,
        eval_data=eval_dataset
    )





if __name__ == "__main__":
    main()