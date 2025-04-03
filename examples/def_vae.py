from itertools import chain
from typing import Union
from random import shuffle, seed

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
MODE = "train_chkp"

CONFIG = {
    "encoder": "bert-base-cased",
    # "encoder": "google/flan-t5-base",
    # "encoder": "NovaSearch/stella_en_1.5B_v5",
    # "encoder": "intfloat/e5-large-v2",
    # "decoder": "gpt2",
    # "decoder": "meta-llama/Llama-3.2-3B",
    "decoder": "meta-llama/Llama-3.1-8B",
    # "decoder": "Qwen/Qwen2.5-1.5B",
    # "decoder": "Qwen/Qwen2.5-3B",
    # "decoder": "mistralai/Mistral-7B-v0.3",
    # "decoder": "microsoft/phi-4",
    "latent_size": 128,
    "max_sent_len": 32,
    "ds_prefix": "wkt_wn_eb",
    "num_epochs": 20,
    "batch_size": 10 if (MODE == "dev") else 200,
    "lr": 1e-3,
    "start_beta": 1.0,
    "max_beta": 1.0
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
        seed(0)
        if ("wkt" in config["ds_prefix"]):
            wkt_dataset = WiktionaryDefinitionCorpus.from_resource("pos+lemma+ctag+dep+dsr")
            wkt_dataset = [sent for sent in wkt_dataset if not exclude_sentence(sent)]
            shuffle(wkt_dataset)
            datasets.append(wkt_dataset)
        if ("wn" in config["ds_prefix"]):
            wn_dataset = WordNetFilteredDataSet()
            shuffle(wn_dataset.data)
            datasets.append(wn_dataset)
        if ("eb" in config["ds_prefix"]):
            eb_dataset = EntailmentBankDataSet.from_resource("pos+lemma+ctag+dep+srl#expl_only-noreps")
            shuffle(eb_dataset.data)
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
        model = LangVAE.load_from_folder("wkt_wn_eb-langvae-bert-base-cased-meta-llama__Llama-3.1-8B-l128/VAE_training_2025-03-30_17-54-15/checkpoint_epoch_20")
        model.encoder.to(DEVICE)
        model.decoder.to(DEVICE)
        model.encoder.init_pretrained_model()
        model.decoder.init_pretrained_model()
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
        steps_saving=5,
        optimizer_cls="AdamW",
        # optimizer_params={"weight_decay": 0.05, "betas": (0.91, 0.995)},
        scheduler_cls="ReduceLROnPlateau",
        scheduler_params={"patience": 5, "factor": 0.5},
        start_beta=config["start_beta"],
        max_beta=config["max_beta"],
        n_cycles=int(config["num_epochs"] * 0.8),
        target_kl=2.0,
        # keep_best_on_train=True
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