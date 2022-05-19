import logging
import math

from detectron2.engine import HookBase
from fvcore.common.checkpoint import Checkpointer


class BestCheckpointer(HookBase):
    """
    Checkpoints best weights based off given metric.
    This hook should be used in conjunction to and executed after `EvalHook`.
    """

    def __init__(
            self,
            eval_period: int,
            checkpointer: Checkpointer,
            val_metric: str = "segm/AP50",
            file_prefix: str = "model_best",
            append_iter: bool = False,
    ) -> None:
        """
        Args:
            eval_period (int): the period `EvalHook` is set to run.
            checkpointer: the checkpointer object used to save checkpoints.
            val_metric (str): chosen validation metric to track for best checkpointing
            file_prefix (str): the prefix of checkpoint's filename, defaults to "model_best"
            append_iter (bool): if True, checkpoint file name will have iteration num appended, else write to the same checkpoint file.
        """
        self._logger = logging.getLogger(__name__)
        self._period = eval_period
        self._val_metric = val_metric
        self._checkpointer = checkpointer
        self._file_prefix = file_prefix
        self.best_metric = None
        self.best_iter = None
        self._append_iter = append_iter

    def _best_checking(self):
        latest_metric, metric_iter = self.trainer.storage.latest().get(self._val_metric)
        if self.best_metric is None or latest_metric > self.best_metric:
            if self._append_iter:
                additional_state = {"iteration": metric_iter}
                save_name = f"{self._file_prefix}_{metric_iter:07d}"
            else:
                save_name = f"{self._file_prefix}"
                additional_state = {}
            self._checkpointer.save(save_name, **additional_state)
            if self.best_metric is None:
                self._logger.info(
                    f"Saved first model with latest eval score for {self._val_metric} at {latest_metric:0.5f}"
                )
            else:
                self._logger.info(
                    f"Saved best model as latest eval score for {self._val_metric} at {latest_metric:0.5f} is better than last best score at {self.best_metric:0.5f} @ {self.best_iter} steps"
                )
            if math.isnan(latest_metric):
                latest_metric = -1.0
            self.best_metric = latest_metric
            self.best_iter = metric_iter
        else:
            self._logger.info(
                f"Not saving as latest eval score for {self._val_metric} at {latest_metric:0.5f} is not better than best score at {self.best_metric:0.5f} @ {self.best_iter} steps"
            )

    def after_step(self):
        # same conditions as `EvalHook`
        next_iter = self.trainer.iter + 1
        if (
                self._period > 0
                and next_iter % self._period == 0
                and next_iter != self.trainer.max_iter
        ):
            self._best_checking()

    def after_train(self):
        # same conditions as `EvalHook`
        if self.trainer.iter + 1 >= self.trainer.max_iter:
            self._best_checking()
