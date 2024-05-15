from itertools import chain
from typing import Union

from pythae.models.vae import VAEConfig
from saf_datasets import WordNetFilteredDataSet, WiktionaryDefinitionCorpus
from saf_datasets import EntailmentBankDataSet
from saf import Sentence
from langvae import LangCVAE
from langvae.encoders import AnnotatedSentenceEncoder
from langvae.decoders import SentenceDecoder
from langvae.data_conversion.tokenization import TokenizedAnnotatedDataSet
from langvae.pipelines import LanguageTrainingPipeline
from langvae.trainers import CyclicalScheduleKLThresholdTrainer, CyclicalScheduleKLThresholdTrainerConfig

DEVICE = "cuda"
LATENT_SIZE = 32
MAX_SENT_LEN = 32

def exclude_sentence(sent: Union[Sentence, str]):
    sent_str = sent.surface if (isinstance(sent, Sentence)) else sent
    return sent_str.startswith("plural") or "surname" in sent_str


def main():
    # dataset = [sent for sent in dataset if not exclude_sentence(sent)]
    dataset = [sent for sent in EntailmentBankDataSet.from_resource("pos+lemma+ctag+dep+srl#noproof")
               if (sent.annotations["type"] == "answer" or sent.annotations["type"].startswith("context"))]
    for sent in dataset:
        for token in sent.tokens:
            token.annotations["srl_0"] = token.annotations["srl"][0
            ]
    eval_size = int(0.05 * len(dataset))
    decoder = SentenceDecoder("gpt2", LATENT_SIZE, MAX_SENT_LEN, device=DEVICE)
    encoder = AnnotatedSentenceEncoder("bert-base-cased", LATENT_SIZE, decoder.tokenizer, 1, device=DEVICE)
    train_dataset = TokenizedAnnotatedDataSet(dataset[:-eval_size], decoder.tokenizer, decoder.max_len,
                                              annotations=["srl_0"], device=DEVICE)
    eval_dataset = TokenizedAnnotatedDataSet(dataset[-eval_size:], decoder.tokenizer, decoder.max_len,
                                             annotations=["srl_0"], device=DEVICE)

    encoder.debug = True
    decoder.debug = True

    model_config = VAEConfig(
        input_dim=(train_dataset[0]["data"].shape[-2], train_dataset[0]["data"].shape[-1] * (1 + len(train_dataset.annotations))),
        latent_dim=LATENT_SIZE
    )

    print("Training new model...")
    model = LangCVAE(model_config, encoder, decoder)

    training_config = CyclicalScheduleKLThresholdTrainerConfig(
        output_dir='eb_superv_expl_cvae',
        num_epochs=10,
        learning_rate=1e-4,
        per_device_train_batch_size=50,
        per_device_eval_batch_size=50,
        steps_saving=1,
        optimizer_cls="AdamW",
        # optimizer_params={"weight_decay": 0.05, "betas": (0.91, 0.995)},
        scheduler_cls="ReduceLROnPlateau",
        scheduler_params={"patience": 5, "factor": 0.5},
        max_beta=1.0,
        n_cycles=8,
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

    model.push_to_hf_hub("neuro-symbolic-ai/entailbank_langcvae")





if __name__ == "__main__":
    main()