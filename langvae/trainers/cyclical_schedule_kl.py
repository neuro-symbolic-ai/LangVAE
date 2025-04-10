import os
import logging
import torch
import torch.distributed as dist
from typing import List, Optional
from copy import deepcopy
from torch import Tensor
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from pydantic.dataclasses import dataclass
from pythae.trainers.base_trainer import BaseTrainer, BaseTrainerConfig
from pythae.trainers.training_callbacks import TrainingCallback
from pythae.data.datasets import BaseDataset
from langvae.arch.vae import LangVAE
from langvae.data_conversion.tokenization import collate_sparse_fn

logger = logging.getLogger(__name__)
console = logging.StreamHandler()
logger.addHandler(console)
logger.setLevel(logging.INFO)


def frange_cycle_zero_linear(n_iter: int,
                             start: float = 0.0,
                             stop: float = 1.0,
                             n_cycle: int = 4,
                             ratio_increase: float = 0.5,
                             ratio_zero: float = 0.3) -> Tensor:
    beta_t_list = torch.ones(n_iter) * stop
    period = n_iter/n_cycle
    step = (stop-start)/(period*ratio_increase)  # linear schedule

    for c in range(n_cycle):
        v, i = start, 0
        while v <= stop and (int(i+c*period) < n_iter):
            if i < period*ratio_zero:
                beta_t_list[int(i+c*period)] = start
            else:
                beta_t_list[int(i+c*period)] = v
                v += step
            i += 1
    return beta_t_list


def copy_model_ptref(model: LangVAE) -> LangVAE:
    pt_encoder = model.encoder._encoder
    pt_enc_tokenizer = model.encoder._tokenizer
    pt_decoder = model.decoder._decoder
    pt_dec_tokenizer = model.decoder._tokenizer

    model.encoder._encoder = None
    model.encoder._tokenizer = None
    model.decoder._decoder = None
    model.decoder._tokenizer = None

    model_copy = deepcopy(model)

    model.encoder._encoder = pt_encoder
    model.encoder._tokenizer = pt_enc_tokenizer
    model.decoder._decoder = pt_decoder
    model.decoder._tokenizer = pt_dec_tokenizer

    model_copy.encoder._encoder = pt_encoder
    model_copy.encoder._tokenizer = pt_enc_tokenizer
    model_copy.decoder._decoder = pt_decoder
    model_copy.decoder._tokenizer = pt_dec_tokenizer

    return model_copy



@dataclass
class CyclicalScheduleKLThresholdTrainerConfig(BaseTrainerConfig):
    """
    Extended training config class for the CyclicalScheduleKLThresholdTrainer

    Attributes:
        start_beta (float): Initial value of `beta`.
        max_beta (float): Maximum value of `beta`.
        n_cycles (int): Total number of cycles for the beta annealing.
        target_kl (float): Target KL threshold (minimum), after which KL loss is zeroed
        for any given dimension below the threshold.
    """
    start_beta: float = 0.0
    max_beta: float = 1.0
    n_cycles: int = 1
    target_kl: float = 2.0


class CyclicalScheduleKLThresholdTrainer(BaseTrainer):
    """Class to perform model training with cyclical schedule KL threshold (beta annealing).

    Args:
        model (LangVAE): A instance of :class:`~langvae.arch.vae.LangVAE` to train

        train_dataset (BaseDataset): The training dataset of type
            :class:`~pythae.data.dataset.BaseDataset`

        eval_dataset (BaseDataset): The evaluation dataset of type
            :class:`~pythae.data.dataset.BaseDataset`

        training_config (CyclicalScheduleKLThresholdTrainerConfig): The training arguments summarizing the main
            parameters used for training. If None, a basic training instance of
            :class:`CyclicalScheduleKLThresholdTrainerConfig` is used. Default: None.

        callbacks (List[~pythae.trainers.training_callbacks.TrainingCallbacks]):
            A list of callbacks to use during training.
    """

    def __init__(
            self,
            model: LangVAE,
            train_dataset: BaseDataset,
            eval_dataset: Optional[BaseDataset] = None,
            training_config: Optional[CyclicalScheduleKLThresholdTrainerConfig] = None,
            callbacks: List[TrainingCallback] = None,
    ):
        if training_config is None:
            training_config = CyclicalScheduleKLThresholdTrainerConfig()

        super().__init__(model, train_dataset, eval_dataset, training_config, callbacks)

        self.num_batches = len(train_dataset) // training_config.per_device_train_batch_size
        self.num_batches += len(train_dataset) % training_config.per_device_train_batch_size
        n_iter = training_config.num_epochs * self.num_batches
        self.beta_t_list = frange_cycle_zero_linear(n_iter, start=training_config.start_beta, stop=training_config.max_beta,
                                                    n_cycle=training_config.n_cycles, ratio_increase=0.25, ratio_zero=0.25)
        self.model.target_kl = training_config.target_kl

    def train_step(self, epoch: int):
        """The trainers performs training loop over the train_loader.

        Parameters:
            epoch (int): The current epoch number

        Returns:
            (torch.Tensor): The step training loss
        """
        self.callback_handler.on_train_step_begin(
            training_config=self.training_config,
            train_loader=self.train_loader,
            epoch=epoch,
            rank=self.rank,
        )

        # set model in train model
        self.model.train()

        epoch_loss = 0

        for i, inputs in enumerate(self.train_loader):

            inputs = self._set_inputs_to_device(inputs)
            self.model.cur_beta = self.beta_t_list[i + (epoch - 1) * self.num_batches]

            with self.amp_context:
                model_output = self.model(
                    inputs,
                    epoch=epoch,
                    dataset_size=len(self.train_loader.dataset),
                    uses_ddp=self.distributed,
                )

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

            self._optimizers_step(model_output)

            loss = model_output.loss

            epoch_loss += loss.item()

            if epoch_loss != epoch_loss:
                raise ArithmeticError("NaN detected in train loss")

            self.callback_handler.on_train_step_end(
                training_config=self.training_config
            )

        # Allows model updates if needed
        if self.distributed:
            self.model.module.update()
        else:
            self.model.update()

        epoch_loss /= len(self.train_loader)

        return epoch_loss

    def eval_step(self, epoch: int):
        """Perform an evaluation step

        Parameters:
            epoch (int): The current epoch number

        Returns:
            (torch.Tensor): The evaluation loss
        """

        self.callback_handler.on_eval_step_begin(
            training_config=self.training_config,
            eval_loader=self.eval_loader,
            epoch=epoch,
            rank=self.rank,
        )

        self.model.eval()
        # cur_beta = self.model.cur_beta
        # self.model.cur_beta = 1.0

        epoch_loss = 0

        with self.amp_context:
            for inputs in self.eval_loader:

                inputs = self._set_inputs_to_device(inputs)

                try:
                    with torch.no_grad():

                        model_output = self.model(
                            inputs,
                            epoch=epoch,
                            dataset_size=len(self.eval_loader.dataset),
                            uses_ddp=self.distributed,
                        )

                except RuntimeError:
                    model_output = self.model(
                        inputs,
                        epoch=epoch,
                        dataset_size=len(self.eval_loader.dataset),
                        uses_ddp=self.distributed,
                    )

                loss = model_output.loss

                epoch_loss += loss.item()

                if epoch_loss != epoch_loss:
                    raise ArithmeticError("NaN detected in eval loss")

                self.callback_handler.on_eval_step_end(
                    training_config=self.training_config
                )

        epoch_loss /= len(self.eval_loader)

        # self.model.cur_beta = cur_beta
        # if (self.model.debug):
        #     LangVAE.loss_writer.add_scalar("Loss/eval", epoch_loss, self.model._dbg_counter // 10)
        #     LangVAE.loss_writer.flush()

        return epoch_loss

    def get_train_dataloader(
        self, train_dataset: BaseDataset
    ) -> torch.utils.data.DataLoader:
        if self.distributed:
            train_sampler = DistributedSampler(
                train_dataset, num_replicas=self.world_size, rank=self.rank
            )
        else:
            train_sampler = None
        return DataLoader(
            dataset=train_dataset,
            batch_size=self.training_config.per_device_train_batch_size,
            num_workers=self.training_config.train_dataloader_num_workers,
            shuffle=False, #(train_sampler is None),
            sampler=train_sampler,
            collate_fn=collate_sparse_fn,
        )

    def get_eval_dataloader(
        self, eval_dataset: BaseDataset
    ) -> torch.utils.data.DataLoader:
        if self.distributed:
            eval_sampler = DistributedSampler(
                eval_dataset, num_replicas=self.world_size, rank=self.rank
            )
        else:
            eval_sampler = None
        return DataLoader(
            dataset=eval_dataset,
            batch_size=self.training_config.per_device_eval_batch_size,
            num_workers=self.training_config.eval_dataloader_num_workers,
            shuffle=(eval_sampler is None),
            sampler=eval_sampler,
            collate_fn=collate_sparse_fn,
        )

    def train(self, log_output_dir: str = None):
        """This function is the main training function

        Args:
            log_output_dir (str): The path in which the log will be stored
        """

        self.prepare_training()

        self.callback_handler.on_train_begin(
            training_config=self.training_config, model_config=self.model_config
        )

        log_verbose = False

        msg = (
            f"Training params:\n - max_epochs: {self.training_config.num_epochs}\n"
            " - per_device_train_batch_size: "
            f"{self.training_config.per_device_train_batch_size}\n"
            " - per_device_eval_batch_size: "
            f"{self.training_config.per_device_eval_batch_size}\n"
            f" - checkpoint saving every: {self.training_config.steps_saving}\n"
            f"Optimizer: {self.optimizer}\n"
            f"Scheduler: {self.scheduler}\n"
        )

        if self.is_main_process:
            logger.info(msg)

        # set up log file
        if log_output_dir is not None and self.is_main_process:
            log_verbose = True
            file_logger = self._get_file_logger(log_output_dir=log_output_dir)

            file_logger.info(msg)

        if self.is_main_process:
            logger.info("Successfully launched training !\n")

        # set best losses for early stopping
        best_train_loss = 1e10
        best_eval_loss = 1e10

        for epoch in range(1, self.training_config.num_epochs + 1):

            self.callback_handler.on_epoch_begin(
                training_config=self.training_config,
                epoch=epoch,
                train_loader=self.train_loader,
                eval_loader=self.eval_loader,
            )

            metrics = {}

            epoch_train_loss = self.train_step(epoch)
            metrics["train_epoch_loss"] = epoch_train_loss

            if self.eval_dataset is not None:
                epoch_eval_loss = self.eval_step(epoch)
                metrics["eval_epoch_loss"] = epoch_eval_loss
                self._schedulers_step(epoch_eval_loss)

            else:
                epoch_eval_loss = best_eval_loss
                self._schedulers_step(epoch_train_loss)

            if (
                epoch_eval_loss < best_eval_loss
                and not self.training_config.keep_best_on_train
            ):
                best_eval_loss = epoch_eval_loss
                best_model = copy_model_ptref(self.model)
                self._best_model = best_model

            elif (
                epoch_train_loss < best_train_loss
                and self.training_config.keep_best_on_train
            ):
                best_train_loss = epoch_train_loss
                best_model = copy_model_ptref(self.model)
                self._best_model = best_model

            if (
                self.training_config.steps_predict is not None
                and epoch % self.training_config.steps_predict == 0
                and self.is_main_process
            ):
                true_data, reconstructions, generations = self.predict(best_model)

                self.callback_handler.on_prediction_step(
                    self.training_config,
                    true_data=true_data,
                    reconstructions=reconstructions,
                    generations=generations,
                    global_step=epoch,
                )

            self.callback_handler.on_epoch_end(training_config=self.training_config)

            # save checkpoints
            if (
                self.training_config.steps_saving is not None
                and epoch % self.training_config.steps_saving == 0
            ):
                if self.is_main_process:
                    self.save_checkpoint(
                        model=best_model, dir_path=self.training_dir, epoch=epoch
                    )
                    logger.info(f"Saved checkpoint at epoch {epoch}\n")

                    if log_verbose:
                        file_logger.info(f"Saved checkpoint at epoch {epoch}\n")

            self.callback_handler.on_log(
                self.training_config,
                metrics,
                logger=logger,
                global_step=epoch,
                rank=self.rank,
            )

        final_dir = os.path.join(self.training_dir, "final_model")

        if self.is_main_process:
            self.save_model(best_model, dir_path=final_dir)

            logger.info("Training ended!")
            logger.info(f"Saved final model in {final_dir}")

        if self.distributed:
            dist.destroy_process_group()

        self.callback_handler.on_train_end(self.training_config)
