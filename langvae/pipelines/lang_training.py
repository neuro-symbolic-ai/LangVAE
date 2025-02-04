import logging
import torch
import numpy as np
from typing import List, Optional, Union
from pythae.pipelines.training import TrainingPipeline, TrainingCallback
from pythae.data.preprocessors import BaseDataset
from pythae.customexception import DatasetError
from pythae.trainers import *
from langvae.trainers import CyclicalScheduleKLThresholdTrainer, CyclicalScheduleKLThresholdTrainerConfig
from langvae.data_conversion.tokenization import collate_sparse_fn
logger = logging.getLogger(__name__)

# make it print to the console.
console = logging.StreamHandler()
logger.addHandler(console)
logger.setLevel(logging.INFO)


class LanguageTrainingPipeline(TrainingPipeline):
    def _check_dataset(self, dataset: BaseDataset):

        try:
            dataset_output = dataset[0]

        except Exception as e:
            raise DatasetError(
                "Error when trying to collect data from the dataset. Check `__getitem__` method. "
                "The Dataset should output a dictionnary with keys at least ['data']. "
                "Please check documentation.\n"
                f"Exception raised: {type(e)} with message: " + str(e)
            ) from e

        if "data" not in dataset_output.keys():
            raise DatasetError(
                "The Dataset should output a dictionnary with keys ['data']"
            )

        try:
            len(dataset)

        except Exception as e:
            raise DatasetError(
                "Error when trying to get dataset len. Check `__len__` method. "
                "Please check documentation.\n"
                f"Exception raised: {type(e)} with message: " + str(e)
            ) from e

        # check everything if fine when combined with data loader
        from torch.utils.data import DataLoader

        dataloader = DataLoader(
            dataset=dataset,
            batch_size=min(len(dataset), 2),
            collate_fn=collate_sparse_fn,
        )
        loader_out = next(iter(dataloader))
        assert len(loader_out.data) == min(
            len(dataset), 2
        ), "Error when combining dataset with loader."

    def __call__(
        self,
        train_data: Union[np.ndarray, torch.Tensor, torch.utils.data.Dataset],
        eval_data: Union[np.ndarray, torch.Tensor, torch.utils.data.Dataset] = None,
        callbacks: List[TrainingCallback] = None,
    ):
        """
        Launch the model training on the provided data.

        Args:
            training_data (Union[~numpy.ndarray, ~torch.Tensor]): The training data as a
                :class:`numpy.ndarray` or :class:`torch.Tensor` of shape (mini_batch x
                n_channels x ...)

            eval_data (Optional[Union[~numpy.ndarray, ~torch.Tensor]]): The evaluation data as a
                :class:`numpy.ndarray` or :class:`torch.Tensor` of shape (mini_batch x
                n_channels x ...). If None, only uses train_fata for training. Default: None.

            callbacks (List[~pythae.trainers.training_callbacks.TrainingCallbacks]):
                A list of callbacks to use during training.
        """

        if isinstance(train_data, np.ndarray) or isinstance(train_data, torch.Tensor):

            logger.info("Preprocessing train data...")
            train_data = self.data_processor.process_data(train_data)
            train_dataset = self.data_processor.to_dataset(train_data)

        else:
            train_dataset = train_data

        logger.info("Checking train dataset...")
        self._check_dataset(train_dataset)

        if eval_data is not None:
            if isinstance(eval_data, np.ndarray) or isinstance(eval_data, torch.Tensor):
                logger.info("Preprocessing eval data...\n")
                eval_data = self.data_processor.process_data(eval_data)
                eval_dataset = self.data_processor.to_dataset(eval_data)

            else:
                eval_dataset = eval_data

            logger.info("Checking eval dataset...")
            self._check_dataset(eval_dataset)

        else:
            eval_dataset = None

        if isinstance(self.training_config, CoupledOptimizerTrainerConfig):
            logger.info("Using Coupled Optimizer Trainer\n")
            trainer = CoupledOptimizerTrainer(
                model=self.model,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                training_config=self.training_config,
                callbacks=callbacks,
            )

        elif isinstance(self.training_config, AdversarialTrainerConfig):
            logger.info("Using Adversarial Trainer\n")
            trainer = AdversarialTrainer(
                model=self.model,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                training_config=self.training_config,
                callbacks=callbacks,
            )

        elif isinstance(self.training_config, CoupledOptimizerAdversarialTrainerConfig):
            logger.info("Using Coupled Optimizer Adversarial Trainer\n")
            trainer = CoupledOptimizerAdversarialTrainer(
                model=self.model,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                training_config=self.training_config,
                callbacks=callbacks,
            )

        elif isinstance(self.training_config, CyclicalScheduleKLThresholdTrainerConfig):
            logger.info("Using Base Trainer\n")
            trainer = CyclicalScheduleKLThresholdTrainer(
                model=self.model,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                training_config=self.training_config,
                callbacks=callbacks,
            )

        elif isinstance(self.training_config, BaseTrainerConfig):
            logger.info("Using Base Trainer\n")
            trainer = BaseTrainer(
                model=self.model,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                training_config=self.training_config,
                callbacks=callbacks,
            )
        else:
            raise ValueError("The provided training config is not supported.")

        self.trainer = trainer

        trainer.train()
