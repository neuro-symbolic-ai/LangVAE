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
from langvae import LangVAE
from langvae.encoders import SentenceEncoder
from langvae.decoders import SentenceDecoder
from langvae.data_conversion.tokenization import TokenizedDataSet
from langvae.pipelines import LanguageTrainingPipeline
from langvae.trainers import CyclicalScheduleKLThresholdTrainerConfig
from saf_datasets import EntailmentBankDataSet

DEVICE = "cuda"
LATENT_SIZE = 32
MAX_SENT_LEN = 32

# Load pre-trained sentence encoder and decoder models.
decoder = SentenceDecoder("gpt2", LATENT_SIZE, MAX_SENT_LEN, device=DEVICE)
encoder = SentenceEncoder("bert-base-cased", LATENT_SIZE, decoder.tokenizer, device=DEVICE)

# Select explanatory sentences from the EntailmentBank dataset.
dataset = [
    sent for sent in EntailmentBankDataSet()
    if (sent.annotations["type"] == "answer" or 
        sent.annotations["type"].startswith("context"))
]

# Set training and evaluation datasets with auto tokenization.
eval_size = int(0.1 * len(dataset))
train_dataset = TokenizedDataSet(dataset[:-eval_size], decoder.tokenizer, decoder.max_len)
eval_dataset = TokenizedDataSet(dataset[-eval_size:], decoder.tokenizer, decoder.max_len)


# Define VAE model configuration
model_config = VAEConfig(
    input_dim=(train_dataset[0]["data"].shape[-2], train_dataset[0]["data"].shape[-1]),
    latent_dim=LATENT_SIZE
)

# Initialize LangVAE model
model = LangVAE(model_config, encoder, decoder)

# Train VAE on explanatory sentences
training_config = CyclicalScheduleKLThresholdTrainerConfig(
    output_dir='expl_vae',
    num_epochs=5,
    learning_rate=1e-4,
    per_device_train_batch_size=50,
    per_device_eval_batch_size=50,
    steps_saving=1,
    optimizer_cls="AdamW",
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
```

This example loads pre-trained encoder and decoder models, defines a VAE model configuration, initializes the LangVAE model, and trains it on text data using a custom training pipeline.


## License

LangVAE is licensed under the GPLv3 License. See the LICENSE file for details.
