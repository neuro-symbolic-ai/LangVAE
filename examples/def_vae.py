from itertools import chain
from typing import Union

import torch.cuda
from tqdm import tqdm
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
from langvae.trainers.training_callbacks import TensorBoardCallback

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODE = "train"

CONFIG = {
    "encoder": "bert-base-cased",
    "decoder": "gpt2",
    # "encoder": "google/flan-t5-base",
    # "decoder": "meta-llama/Llama-3.2-3B",
    "latent_size": 128,
    "max_sent_len": 32,
    "ds_prefix": "wkt",
    "num_epochs": 50,
    "batch_size": 10 if (MODE == "dev") else 200,
    "lr": 1e-3,
    "max_beta": 0.2
}

torch.set_float32_matmul_precision('high')

def exclude_sentence(sent: Union[Sentence, str]):
    sent_str = sent.surface if (isinstance(sent, Sentence)) else sent
    return sent_str.startswith("plural") or "surname" in sent_str


def main(config: dict):
    if (MODE == "dev"):
        datasets = [WordNetFilteredDataSet()[:1000],]
    else:
        datasets = list()
        if ("wkt" in config["ds_prefix"]):
            wkt_dataset = WiktionaryDefinitionCorpus.from_resource("pos+lemma+ctag+dep+dsr")
            wkt_dataset = [sent for sent in wkt_dataset if not exclude_sentence(sent)]
            datasets.append(wkt_dataset)
        if ("wn" in config["ds_prefix"]):
            wn_dataset = WordNetFilteredDataSet()
            datasets.append(wn_dataset)
        if ("eb" in config["ds_prefix"]):
            eb_dataset = [sent for sent in EntailmentBankDataSet()
                          if (sent.annotations["type"] == "answer" or sent.annotations["type"].startswith("context"))]
            datasets.append(eb_dataset)

    # eval_size = int(0.05 * len(dataset))
    eval_size = [int(0.01 * len(ds)) for ds in datasets]
    latent_size = config["latent_size"]
    max_sent_len = config["max_sent_len"]
    ds_prefix = config["ds_prefix"]

    decoder = SentenceDecoder(config["decoder"], latent_size, max_sent_len, device=DEVICE, device_map="auto")
    # decoder = SentenceDecoder("princeton-nlp/Sheared-LLaMA-2.7B", LATENT_SIZE, MAX_SENT_LEN, device=DEVICE,
    #                           load_in_4bit=True, device_map="auto")
    encoder = SentenceEncoder(config["encoder"], latent_size, decoder.tokenizer, caching=True, device=DEVICE)
    train_dataset = TokenizedDataSet(sorted(chain(*[datasets[i][:-eval_size[i]] for i in range(len(datasets))]),
                                            key=lambda x: len(x.surface), reverse=True),
                                     decoder.tokenizer, decoder.max_len, caching=True,
                                     cache_persistence=f"{ds_prefix}_train_tok-{config['decoder']}_cache.jsonl")
    eval_dataset = TokenizedDataSet(sorted(chain(*[datasets[i][-eval_size[i]:] for i in range(len(datasets))]),
                                           key=lambda x: len(x.surface), reverse=True),
                                    decoder.tokenizer, decoder.max_len, caching=True,
                                    cache_persistence=f"{ds_prefix}_eval_tok-{config['decoder']}_cache.jsonl")

    encoder.debug = True
    decoder.debug = True

    model_config = VAEConfig(
        # input_dim=(train_dataset[0]["data"].shape[-2], train_dataset[0]["data"].shape[-1]),
        latent_dim=latent_size
    )

    if (MODE == "train_chkp"):
        print("Loading checkpoint...")
        model = LangVAE.load_from_folder("wn-langvae-bert-base-cased-gpt2-l128/VAE_training_2024-12-20_20-21-52/checkpoint_epoch_40")
        model.encoder.to(DEVICE)
        model.decoder.to(DEVICE)
    else:
        print("Training new model...")
        model = LangVAE(model_config, encoder, decoder)

    # model.debug = True

    exp_label = f"{ds_prefix}-langvae-{config['encoder'].replace('/', '__')}-{config['decoder'].replace('/', '__')}-l{latent_size}"

    training_config = CyclicalScheduleKLThresholdTrainerConfig(
        output_dir=exp_label,
        num_epochs=config["num_epochs"],
        learning_rate=config["lr"],
        per_device_train_batch_size=config["batch_size"],
        per_device_eval_batch_size=config["batch_size"],
        steps_saving=10,
        optimizer_cls="AdamW",
        # optimizer_params={"weight_decay": 0.05, "betas": (0.91, 0.995)},
        scheduler_cls="ReduceLROnPlateau",
        scheduler_params={"patience": 5, "factor": 0.5},
        max_beta=config["max_beta"],
        n_cycles=1,
        target_kl=2.0,
        keep_best_on_train=True
    )

    pipeline = LanguageTrainingPipeline(
        training_config=training_config,
        model=model
    )

    exp_params = f"-lr[{config['lr']}]-bsize[{config['batch_size']}]-max_beta[{config['max_beta']}]"

    tb_callback = TensorBoardCallback(exp_label + exp_params)

    pipeline(
        train_data=train_dataset,
        eval_data=eval_dataset,
        callbacks=[tb_callback]
    )

    # LangVAE.loss_writer.close()
    # model.push_to_hf_hub(f"neuro-symbolic-ai/{ds_prefix}_langvae_{config['encoder']}_{config['decoder']}_l{latent_size}")



if __name__ == "__main__":
    main(CONFIG)