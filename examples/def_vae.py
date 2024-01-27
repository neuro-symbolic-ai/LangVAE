from pythae.pipelines import TrainingPipeline
from pythae.models import VAEConfig
from pythae.trainers import BaseTrainerConfig
from saf_datasets import CPAEDataSet, WiktionaryDefinitionCorpus
from langvae.arch.vae import LangVAE
from langvae.encoders import SentenceEncoder
from langvae.decoders import SentenceDecoder
from langvae.data_conversion.tokenization import TokenizedDataSet

DEVICE = "cpu"
LATENT_SIZE = 32
MAX_SENT_LEN = 32


def main():
    dataset = WiktionaryDefinitionCorpus.from_resource("pos+lemma+ctag+dep+dsr")
    decoder = SentenceDecoder("gpt2", LATENT_SIZE, MAX_SENT_LEN, device=DEVICE)
    # decoder = SentenceDecoder("princeton-nlp/Sheared-LLaMA-2.7B", LATENT_SIZE, MAX_SENT_LEN, device=DEVICE,
    #                           load_in_4bit=True, device_map="auto")
    encoder = SentenceEncoder("bert-base-cased", LATENT_SIZE, decoder.tokenizer, device=DEVICE)
    train_dataset = TokenizedDataSet(dataset[:-1000], decoder.tokenizer, decoder.max_len, device=DEVICE)
    eval_dataset = TokenizedDataSet(dataset[-1000:], decoder.tokenizer, decoder.max_len, device=DEVICE)

    encoder.debug = True
    decoder.debug = True

    model_config = VAEConfig(
        input_dim=(train_dataset[0]["data"].shape[-2], train_dataset[0]["data"].shape[-1]),
        latent_dim=LATENT_SIZE
    )
    model = LangVAE(model_config, encoder, decoder)

    training_config = BaseTrainerConfig(
        output_dir='def_vae',
        num_epochs=10,
        learning_rate=1e-4,
        per_device_train_batch_size=20,
        per_device_eval_batch_size=20,
        steps_saving=1,
        optimizer_cls="AdamW",
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