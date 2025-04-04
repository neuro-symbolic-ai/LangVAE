API
===

.. toctree::
    :maxdepth: 2
    :caption: Packages
    :titlesonly:

    langvae/langvae.arch
    langvae/langvae.encoders
    langvae/langvae.decoders
    langvae/langvae.data_conversion
    langvae/langvae.trainers
    langvae/langvae.pipelines


Core classes
-------------------------------------------

.. autosummary::
    ~langvae.arch.vae.LangVAE
    ~langvae.encoders.sentence.SentenceEncoder
    ~langvae.decoders.sentence.SentenceDecoder
    ~langvae.data_conversion.tokenization.TokenizedDataSet
    ~langvae.data_conversion.tokenization.TokenizedAnnotatedDataSet
    :nosignatures:



Training pipeline
-------------------------------------------

.. autosummary::
    ~langvae.trainers.cyclical_schedule_kl.CyclicalScheduleKLThresholdTrainer
    ~langvae.trainers.cyclical_schedule_kl.CyclicalScheduleKLThresholdTrainerConfig
    ~langvae.pipelines.lang_training.LanguageTrainingPipeline
    ~langvae.trainers.training_callbacks.TensorBoardCallback
    :nosignatures:
