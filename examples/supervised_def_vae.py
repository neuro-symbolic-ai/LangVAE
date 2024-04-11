from itertools import chain
from typing import Union

from pythae.models.vae import VAEConfig
from saf_datasets import WordNetFilteredDataSet, WiktionaryDefinitionCorpus
from saf_datasets import EntailmentBankDataSet
from saf import Sentence
from langvae import LangVAE
from langvae.encoders import AnnotatedSentenceEncoder
from langvae.decoders import AnnotatedSentenceDecoder
from langvae.data_conversion.tokenization import TokenizedAnnotatedDataSet
from langvae.pipelines import LanguageTrainingPipeline
from langvae.trainers import CyclicalScheduleKLThresholdTrainer, CyclicalScheduleKLThresholdTrainerConfig

DEVICE = "cpu"
LATENT_SIZE = 32
MAX_SENT_LEN = 32

def exclude_sentence(sent: Union[Sentence, str]):
    sent_str = sent.surface if (isinstance(sent, Sentence)) else sent
    return sent_str.startswith("plural") or "surname" in sent_str


def main():
    dataset = WiktionaryDefinitionCorpus.from_resource("pos+lemma+ctag+dep+dsr")
    dataset = [sent for sent in dataset if not exclude_sentence(sent)]
    eval_size = int(0.05 * len(dataset))
    decoder = AnnotatedSentenceDecoder("gpt2", LATENT_SIZE, MAX_SENT_LEN, 1, device=DEVICE)
    # decoder = SentenceDecoder("princeton-nlp/Sheared-LLaMA-2.7B", LATENT_SIZE, MAX_SENT_LEN, device=DEVICE,
    #                           load_in_4bit=True, device_map="auto")
    encoder = AnnotatedSentenceEncoder("bert-base-cased", LATENT_SIZE, decoder.tokenizer, 1, device=DEVICE)
    train_dataset = TokenizedAnnotatedDataSet(dataset[:-eval_size], decoder.tokenizer, decoder.max_len,
                                              annotations=["dsr"], device=DEVICE)
    eval_dataset = TokenizedAnnotatedDataSet(dataset[-eval_size:], decoder.tokenizer, decoder.max_len,
                                             annotations=["dsr"], device=DEVICE)

    encoder.debug = True
    decoder.debug = True

    model_config = VAEConfig(
        input_dim=(train_dataset[0]["data"].shape[-2], train_dataset[0]["data"].shape[-1] * (1 + len(train_dataset.annotations))),
        latent_dim=LATENT_SIZE
    )
    # try:
    # print("Loading checkpoint...")
    # model = LangVAE.load_from_folder("def_expl_vae/VAE_training_2024-03-18_20-19-52/checkpoint_epoch_4")
    # except:
    print("Training new model...")
    model = LangVAE(model_config, encoder, decoder)

    training_config = CyclicalScheduleKLThresholdTrainerConfig(
        output_dir='def_expl_vae',
        num_epochs=4,
        learning_rate=1e-4,
        per_device_train_batch_size=50,
        per_device_eval_batch_size=50,
        steps_saving=1,
        optimizer_cls="AdamW",
        # optimizer_params={"weight_decay": 0.05, "betas": (0.91, 0.995)},
        scheduler_cls="ReduceLROnPlateau",
        scheduler_params={"patience": 5, "factor": 0.5},
        max_beta=1.0,
        n_cycles=40,
        target_kl=2.0
    )

    pipeline = LanguageTrainingPipeline(
        training_config=training_config,
        model=model
    )

    pipeline(
        train_data=train_dataset,
        eval_data=eval_dataset
    )





if __name__ == "__main__":
    main()