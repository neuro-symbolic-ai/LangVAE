# LangVAE

LangVAE is a Python library for training and evaluating language models using Variational Autoencoders (VAEs). It provides an easy-to-use interface to train VAEs on text data, allowing users to customize the model architecture, loss function, and training parameters.

## Installation

To install LangVAE, simply run:

```bash
pip install langvae
```

This will install all necessary dependencies and set up the package for use in your Python projects.

## Usage

Here's a basic example of how to train a VAE on text data using LangVAE:

```python
from langvae import LangVAE

# Load pre-trained encoder and decoder models
encoder = SentenceEncoder("bert-base-cased", LATENT_SIZE, decoder.tokenizer, device=DEVICE)
decoder = SentenceDecoder("unsloth/llama-3-8b", LATENT_SIZE, MAX_SENT_LEN, device=DEVICE)

# Define VAE model configuration
model_config = VAEConfig(
    input_dim=(train_dataset[0]["data"].shape[-2], train_dataset[0]["data"].shape[-1]),
    latent_dim=LATENT_SIZE
)

# Initialize LangVAE model
model = LangVAE(model_config, encoder, decoder)

# Train VAE on text data
training_config = CyclicalScheduleKLThresholdTrainerConfig(
    output_dir='def_expl_vae',
    num_epochs=5,
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
```

This example loads pre-trained encoder and decoder models, defines a VAE model configuration, initializes the LangVAE model, and trains it on text data using a custom training pipeline.

## Contributing

Contributions to LangVAE are welcome! If you have any suggestions or bug reports, please open an issue on GitHub. If you'd like to contribute code, feel free to submit a pull request with your changes.

## License

LangVAE is licensed under the MIT License. See the LICENSE file for details.
