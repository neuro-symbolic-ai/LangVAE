from torch.utils.tensorboard import SummaryWriter
from pythae.trainers.training_callbacks import TrainingCallback
from pythae.trainers.base_trainer.base_training_config import BaseTrainerConfig


class TensorBoardCallback(TrainingCallback):  # pragma: no cover
    """
    A :class:`TrainingCallback` integrating the experiment tracking tool
    `tensorboard` (https://www.tensorflow.org/tensorboard).

    It allows users to store their configs, monitor
    their trainings and compare runs through a graphic interface. To be able use this feature
    you will need:

    - the package `tensorboard` installed in your virtual env. If not you can install it with

    .. code-block::

        $ pip install tensorboard
    """

    def __init__(self, label=""):
        self._loss_writer = SummaryWriter(comment="::" + label)
        self._counter = 0

    def on_log(self, training_config: BaseTrainerConfig, logs, **kwargs):
        rank = kwargs.pop("rank", -1)

        if (rank == -1 or rank == 0):
            epoch_train_loss = logs.get("train_epoch_loss", None)
            epoch_eval_loss = logs.get("eval_epoch_loss", None)

            if (epoch_train_loss):
                self._loss_writer.add_scalar("Loss/train_joint", epoch_train_loss, self._counter)
            if (epoch_eval_loss):
                self._loss_writer.add_scalar("Loss/eval_joint", epoch_eval_loss, self._counter)

            self._loss_writer.flush()
            self._counter += 1
