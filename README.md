# LangVAE: Large Language VAEs made simple 

LangVAE is a Python library for training and running language models using Variational Autoencoders (VAEs). It provides an easy-to-use interface to train VAEs on text data, allowing users to customize the model architecture, loss function, and training parameters.

## Installation

To install LangVAE, simply run:

```bash
pip install langvae
```

This will install all necessary dependencies and set up the package for use in your Python projects.

## Usage

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


## License

LangVAE is licensed under the GPLv3 License. See the LICENSE file for details.
