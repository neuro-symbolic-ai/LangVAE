import torch
from typing import List, Optional
from torch import Tensor
from pydantic.dataclasses import dataclass
from pythae.trainers.base_trainer import BaseTrainer, BaseTrainerConfig
from pythae.trainers.training_callbacks import TrainingCallback
from pythae.models.base import BaseAE
from pythae.data.datasets import BaseDataset


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

@dataclass
class CyclicalScheduleKLThresholdTrainerConfig(BaseTrainerConfig):
    max_beta: float = 1.0
    n_cycles: int = 1
    target_kl: float = 2.0


class CyclicalScheduleKLThresholdTrainer(BaseTrainer):
    def __init__(
            self,
            model: BaseAE,
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
        self.beta_t_list = frange_cycle_zero_linear(n_iter, start=0.0, stop=training_config.max_beta,
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