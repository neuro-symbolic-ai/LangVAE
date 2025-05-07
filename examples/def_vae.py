import os
import torch.cuda
from itertools import chain
from typing import Union
from random import shuffle, seed
from tqdm import tqdm
from pythae.models.vae import VAEConfig
from saf_datasets import WordNetFilteredDataSet, WiktionaryDefinitionCorpus
from saf_datasets import EntailmentBankDataSet, AllNLIDataSet, STSBDataSet
from saf import Sentence
from langvae import LangVAE
from langvae.encoders import SentenceEncoder
from langvae.decoders import SentenceDecoder
from langvae.data_conversion.tokenization import TokenizedDataSet
from langvae.pipelines import LanguageTrainingPipeline
from langvae.trainers import CyclicalScheduleKLThresholdTrainer, CyclicalScheduleKLThresholdTrainerConfig
from langvae.trainers.training_callbacks import TensorBoardCallback
from langvae.pipelines.evaluation import eval_model

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODE = "train"

CONFIG = {
    "encoder": "bert-base-cased",
    # "encoder": "google/flan-t5-base",
    # "encoder": "NovaSearch/stella_en_1.5B_v5",
    # "encoder": "intfloat/e5-large-v2",
    "decoder": "gpt2",
    # "decoder": "meta-llama/Llama-3.2-3B",
    # "decoder": "meta-llama/Llama-3.1-8B",
    # "decoder": "Qwen/Qwen2.5-1.5B",
    # "decoder": "Qwen/Qwen2.5-3B",
    # "decoder": "mistralai/Mistral-7B-v0.3",
    # "decoder": "microsoft/phi-4",
    # "decoder": "vandijklab/C2S-Scale-Pythia-1b-pt",
    "latent_size": 128,
    "max_sent_len": 32,
    "mem_factor": 1.0,
    "teacher_forcing": True,
    "ds_prefix": "wkt_wn_anli_stsb_eb",
    "num_epochs": 10,
    "batch_size": 10 if (MODE == "dev") else 200,
    "lr": 1e-3,
    "start_beta": 0.1,
    "max_beta": 0.1,
    "target_kl": 0.1
}

torch.set_float32_matmul_precision('high')

def exclude_sentence(sent: Union[Sentence, str]):
    sent_str = sent.surface if (isinstance(sent, Sentence)) else sent
    return sent_str.startswith("plural") or "surname" in sent_str


def main(config: dict):
    if (MODE == "dev"):
        datasets = [EntailmentBankDataSet.from_resource("pos+lemma+ctag+dep+srl#expl_only-noreps")[:1000],]
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
        if ("anli" in config["ds_prefix"]):
            anli_dataset = [sent for sent in AllNLIDataSet() if (sent.annotations["split"] == "train")]
            shuffle(anli_dataset.data)
            datasets.append(anli_dataset)
        if ("stsb" in config["ds_prefix"]):
            stsb_dataset = [sent for sent in STSBDataSet() if (sent.annotations["split"] == "train")]
            shuffle(stsb_dataset.data)
            datasets.append(stsb_dataset)

    # eval_size = int(0.05 * len(dataset))
    eval_size = [int(0.01 * len(ds)) for ds in datasets]
    latent_size = config["latent_size"]
    max_sent_len = config["max_sent_len"]
    ds_prefix = config["ds_prefix"]

    decoder = SentenceDecoder(config["decoder"], latent_size, max_sent_len, device=DEVICE, device_map="auto",
                              memory_factor=config["mem_factor"], teacher_forcing=config["teacher_forcing"])
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
        checkpoint_dir = "wn_eb-langvae-bert-base-cased-gpt2-l128-m1.0/VAE_training_2025-05-05_16-33-36/final_model"
        model = LangVAE.load_from_folder(checkpoint_dir)
        model.encoder.to(DEVICE)
        model.decoder.to(DEVICE)
        model.encoder.init_pretrained_model()
        model.decoder.init_pretrained_model()
    else:
        print("Training new model...")
        model = LangVAE(model_config, encoder, decoder)

    # model.debug = True

    exp_label = f"{ds_prefix}-langvae-{config['encoder'].replace('/', '__')}-{config['decoder'].replace('/', '__')}-l{latent_size}-m{config['mem_factor']}"

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
        start_beta=config["start_beta"],
        max_beta=config["max_beta"],
        n_cycles=int(config["num_epochs"] * 0.8),
        target_kl=config["target_kl"],
        keep_best_on_train=True
    )

    pipeline = LanguageTrainingPipeline(
        training_config=training_config,
        model=model
    )

    # if (MODE == "train_chkp"):
    #     optimizer_state = torch.load(os.path.join(checkpoint_dir, "optimizer.pt"),
    #                                  map_location=torch.device(encoder.device),
    #                                  weights_only = True)
    #     scheduler_state = torch.load(os.path.join(checkpoint_dir, "scheduler.pt"),
    #                                  map_location=torch.device(encoder.device),
    #                                  weights_only=True)
    #     pipeline.trainer.optimizer.load_state_dict(optimizer_state)
    #     pipeline.trainer.scheduler.load_state_dict(scheduler_state)


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