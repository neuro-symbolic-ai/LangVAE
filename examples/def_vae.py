from itertools import chain
from typing import Union

from pythae.models.vae import VAEConfig
from saf_datasets import WordNetFilteredDataSet, WiktionaryDefinitionCorpus
from saf_datasets import EntailmentBankDataSet
from saf import Sentence
from langvae import LangVAE
from langvae.encoders import SentenceEncoder
from langvae.decoders import SentenceDecoder
from langvae.data_conversion.tokenization import TokenizedDataSet
from langvae.pipelines import LanguageTrainingPipeline
from langvae.trainers import CyclicalScheduleKLThresholdTrainer, CyclicalScheduleKLThresholdTrainerConfig

DEVICE = "cuda"
LATENT_SIZE = 64
MAX_SENT_LEN = 32

def exclude_sentence(sent: Union[Sentence, str]):
    sent_str = sent.surface if (isinstance(sent, Sentence)) else sent
    return sent_str.startswith("plural") or "surname" in sent_str


def main():
    dataset1 = WiktionaryDefinitionCorpus.from_resource("pos+lemma+ctag+dep+dsr")
    dataset1 = [sent for sent in dataset1 if not exclude_sentence(sent)]
    dataset2 = WordNetFilteredDataSet()
    dataset3 = [sent for sent in EntailmentBankDataSet()
               if (sent.annotations["type"] == "answer" or sent.annotations["type"].startswith("context"))]
    datasets = (dataset1, dataset2, dataset3)
    # eval_size = int(0.05 * len(dataset))
    eval_size = [int(0.01 * len(ds)) for ds in datasets]
    decoder = SentenceDecoder("gpt2", LATENT_SIZE, MAX_SENT_LEN, device=DEVICE, device_map="auto", max_look_behind=1)
    # decoder = SentenceDecoder("princeton-nlp/Sheared-LLaMA-2.7B", LATENT_SIZE, MAX_SENT_LEN, device=DEVICE,
    #                           load_in_4bit=True, device_map="auto")
    encoder = SentenceEncoder("bert-base-cased", LATENT_SIZE, decoder.tokenizer, device=DEVICE)
    train_dataset = TokenizedDataSet(list(chain(*[datasets[i][:-eval_size[i]] for i in range(len(datasets))])), decoder.tokenizer, decoder.max_len, device=DEVICE)
    eval_dataset = TokenizedDataSet(list(chain(*[datasets[i][-eval_size[i]:] for i in range(len(datasets))])), decoder.tokenizer, decoder.max_len, device=DEVICE)
    # train_dataset = TokenizedDataSet(dataset[:-eval_size], decoder.tokenizer, decoder.max_len, device=DEVICE)
    # eval_dataset = TokenizedDataSet(dataset[-eval_size:], decoder.tokenizer, decoder.max_len, device=DEVICE)

    encoder.debug = True
    decoder.debug = True

    model_config = VAEConfig(
        input_dim=(train_dataset[0]["data"].shape[-2], train_dataset[0]["data"].shape[-1]),
        latent_dim=LATENT_SIZE
    )
    # try:
    # print("Loading checkpoint...")
    # model = LangVAE.load_from_folder("def_expl_vae/VAE_training_2024-03-18_20-19-52/checkpoint_epoch_4")
    # except:
    print("Training new model...")
    model = LangVAE(model_config, encoder, decoder)

    model.debug = True

    training_config = CyclicalScheduleKLThresholdTrainerConfig(
        output_dir='wkt_eb_wn-langvae-bert-gpt2-l64',
        num_epochs=5,
        learning_rate=2e-3,
        per_device_train_batch_size=500,
        per_device_eval_batch_size=500,
        steps_saving=1,
        optimizer_cls="AdamW",
        # optimizer_params={"weight_decay": 0.05, "betas": (0.91, 0.995)},
        scheduler_cls="ReduceLROnPlateau",
        scheduler_params={"patience": 5, "factor": 0.5},
        max_beta=1.0,
        n_cycles=4,
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

    LangCVAE.loss_writer.close()
    model.push_to_hf_hub("neuro-symbolic-ai/wkt_eb_wn-langvae-bert-gpt2-l64")



if __name__ == "__main__":
    main()