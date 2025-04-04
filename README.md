# LangVAE: Large Language VAEs made simple 

LangVAE is a Python library for training and running language models using Variational Autoencoders (LM-VAEs). It provides an easy-to-use interface to train LM-VAEs on text data, allowing users to customize the model architecture, loss function, and training parameters.

## Why LangVAE

LangVAE aims to address current LM-VAE limitations and facilitate the development of specialised models and experimentation over next-gen LLMs. Its main advantages over previous frameworks are:

 - Modular architecture allows flexible development of different LM-VAE configurations. Flexible composition of base models and bottleneck parametrisations, loss functions, etc.
 - Compatible with most state-of-the-art autoregressive models.
 - Has substantially reduced computational requirements for training, compared to the SOTA LM-VAE (Optimus), with an average parameter reduction of over 95% measured when using decoder models between 3B to 7B parameters.
 - Supports multi-GPU training and inference.

## Installation

To install LangVAE, simply run:

```bash
pip install langvae
```

This will install all necessary dependencies and set up the package for use in your Python projects.

## Quick start

Here's a basic example of how to train a VAE on text data using LangVAE:

```python
from pythae.models.vae import VAEConfig
from saf_datasets import EntailmentBankDataSet
from langvae import LangVAE
from langvae.encoders import SentenceEncoder
from langvae.decoders import SentenceDecoder
from langvae.data_conversion.tokenization import TokenizedDataSet
from langvae.pipelines import LanguageTrainingPipeline
from langvae.trainers import CyclicalScheduleKLThresholdTrainerConfig
from langvae.trainers.training_callbacks import TensorBoardCallback

DEVICE = "cuda"
LATENT_SIZE = 128
MAX_SENT_LEN = 32

# Load pre-trained sentence encoder and decoder models.
decoder = SentenceDecoder("gpt2", LATENT_SIZE, MAX_SENT_LEN, device=DEVICE, device_map="auto")
encoder = SentenceEncoder("bert-base-cased", LATENT_SIZE, decoder.tokenizer, caching=True, device=DEVICE)

# Select explanatory sentences from the EntailmentBank dataset.
dataset = [
    sent for sent in EntailmentBankDataSet()
    if (sent.annotations["type"] == "answer" or
        sent.annotations["type"].startswith("context"))
]

# Set training and evaluation datasets with auto tokenization.
eval_size = int(0.1 * len(dataset))
train_dataset = TokenizedDataSet(sorted(dataset[:-eval_size], key=lambda x: len(x.surface), reverse=True),
                                 decoder.tokenizer, decoder.max_len, caching=True,
                                 cache_persistence=f"eb_train_tok-gpt2_cache.jsonl")
eval_dataset = TokenizedDataSet(sorted(dataset[-eval_size:], key=lambda x: len(x.surface), reverse=True),
                                decoder.tokenizer, decoder.max_len, caching=True,
                                cache_persistence=f"eb_eval_tok-gpt2_cache.jsonl")


# Define VAE model configuration
model_config = VAEConfig(latent_dim=LATENT_SIZE)

# Initialize LangVAE model
model = LangVAE(model_config, encoder, decoder)

exp_label = f"eb-langvae-bert-gpt2-{LATENT_SIZE}"

# Train VAE on explanatory sentences
training_config = CyclicalScheduleKLThresholdTrainerConfig(
    output_dir=exp_label,
    num_epochs=20,
    learning_rate=1e-3,
    per_device_train_batch_size=50,
    per_device_eval_batch_size=50,
    steps_saving=5,
    optimizer_cls="AdamW",
    scheduler_cls="ReduceLROnPlateau",
    scheduler_params={"patience": 5, "factor": 0.5},
    max_beta=1.0,
    n_cycles=16,  # num_epochs * 0.8
    target_kl=2.0,
    keep_best_on_train=True
)

pipeline = LanguageTrainingPipeline(
    training_config=training_config,
    model=model
)

# Monitor the training progress with `tensorboard --logdir=runs &`
tb_callback = TensorBoardCallback(exp_label)

pipeline(
    train_data=train_dataset,
    eval_data=eval_dataset,
    callbacks=[tb_callback]
)
```

This example loads pre-trained encoder and decoder models, defines a VAE model configuration, initializes the LangVAE model, and trains it on text data using a custom training pipeline.


## How to / Tutorial

A step-by-step interactive breakdown of the quick start example can be found on this [Colab notebook](https://colab.research.google.com/drive/1CCFvPWsQU2VX41guHGT2-uFgHogAejDv). You can try and play with it in Colab using one of our [pre-trained models](https://huggingface.co/neuro-symbolic-ai).

## Pre-trained models available

We have made a [list of pre-trained models](https://huggingface.co/neuro-symbolic-ai) models available on HuggingFace Hub, comprising all combinations of the following models:

- Encoders: 
  + BERT ([base-cased](https://huggingface.co/bert-base-cased))
  + Flan-T5 ([base](https://huggingface.co/google/flan-t5-base))
  + Stella ([en-1.5B_v5](https://huggingface.co/NovaSearch/stella_en_1.5B_v5))
- Decoders:
  + GPT-2 ([base](https://huggingface.co/gpt2))
  + Qwen ([2.5-3B](https://huggingface.co/Qwen/Qwen2.5-3B))
  + Llama ([3.2-3B](https://huggingface.co/meta-llama/Llama-3.2-3B))
  + Mistral ([7B-v0.3](https://huggingface.co/mistralai/Mistral-7B-v0.3))

Non-annotated models follow the following naming convention:
```
neuro-symbolic-ai/<dataset>-langvae-<encoder>-<decoder>-l<latent dim>
```

where &lt;dataset&gt; is an underscore separated combination of: **eb** (EntailmentBank), **wkt** (Wiktionary), and **wn** (Wordnet); &lt;encoder&gt; is the HuggingFace name of the encoder model (minus organization name), with the same applying for &lt;decoder&gt;; and &lt;latent dim&gt; being the dimensionality of the latent space. So, for example: 
>neuro-symbolic-ai/eb-langvae-bert-base-cased-Qwen2.5-3B-l128

is a model combining a BERT (base-cased) encoder with a Qwen (2.5-3B) decoder, with a latent dimension of 128, trained on the EntailmentBank dataset.

The naming convention for the annotated model is very similar:
```
neuro-symbolic-ai/<dataset>-langcvae-<encoder>-<decoder>-<annotations>-l<latent dim>
```

where &lt;annotations&gt; is an underscore separated list of annotations included in the input data. For example:
>neuro-symbolic-ai/eb-langcvae-bert-base-cased-Qwen2.5-3B-srl-l128

Is a model with the same combination as the previous example, but trained with (and expecting) semantic role labels (SRL) as additional inputs, through a TokenizedDataset (see documentation).


## Documentation

Usage and API documentation can be found at https://langvae.readthedocs.io.


## License

LangVAE is licensed under the GPLv3 License. See the LICENSE file for details.


## Citation

If you find this work useful or use it in your research, please consider citing us

```bibtex
@inproceedings{carvalho2025langvae,
 author = {Carvalho, Danilo Silva and Zhang, Yingji and Unsworth, Harriet and Freitas, Andre},
 booktitle = {ArXiv},
 editor = {},
 pages = {0--0},
 publisher = {ArXiv},
 title = {LangVAE and LangSpace: Building and Probing for Language Model VAEs},
 volume = {0},
 year = {2025}
}
```
