import torch
from typing import List, Optional
from torch import Tensor
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from pydantic.dataclasses import dataclass
from pythae.trainers.base_trainer import BaseTrainer, BaseTrainerConfig
from pythae.trainers.training_callbacks import TrainingCallback
from pythae.data.datasets import BaseDataset
from langvae.arch.vae import LangVAE
from langvae.data_conversion.tokenization import collate_sparse_fn


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
