# coding=utf-8
# Copyright 2020-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Callbacks to use with the Trainer class and customize the training loop.
"""

import dataclasses
import json
import os.path
import re
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Union
import warnings

import pandas as pd
from typing import Callable

import torch
from torch.utils.tensorboard import SummaryWriter

from .util import get_str_time, StateEnum
from dataclasses import field
import numpy as np
from tqdm.auto import tqdm
from enum import Enum

from .util import IntervalStrategy, SaveStrategy
# from .trainer import TrainingArguments
from .training_args import TrainingArguments
# from .utils import logging
from .reporter import SubReporter
from .util.util_obj import MetricStrategy, EpochProperty


# logger = logging.get_logger(__name__)
def has_length(dataset):
    """
    Checks if the dataset implements __len__() and it doesn't raise an error
    """
    try:
        return len(dataset) is not None
    except TypeError:
        # TypeError: len() of unsized object
        return False


@dataclass
class TrainerState:
    """
    A class containing the [`Trainer`] inner state that will be saved along the model and optimizer when checkpointing
    and passed to the [`TrainerCallback`].

    <Tip>

    In all this class, one step is to be understood as one update step. When using gradient accumulation, one update
    step may require several forward and backward passes: if you use `gradient_accumulation_steps=n`, then one update
    step requires going through *n* batches.

    </Tip>

    Args:
        epoch (`float`, *optional*):
            Only set during training, will represent the epoch the training is at (the decimal part being the
            percentage of the current epoch completed).
        global_step (`int`, *optional*, defaults to 0):
            During training, represents the number of update steps completed.
        max_steps (`int`, *optional*, defaults to 0):
            The number of update steps to do during the current training.
        logging_steps (`int`, *optional*, defaults to 500):
            Log every X updates steps
        eval_steps (`int`, *optional*):
            Run an evaluation every X steps.
        save_steps (`int`, *optional*, defaults to 500):
            Save checkpoint every X updates steps.
        train_batch_size (`int`, *optional*):
            The batch size for the training dataloader. Only needed when
            `auto_find_batch_size` has been used.
        num_input_tokens_seen (`int`, *optional*, defaults to 0):
            When tracking the inputs tokens, the number of tokens seen during training (number of input tokens, not the
            number of prediction tokens).
        total_flos (`float`, *optional*, defaults to 0):
            The total number of floating operations done by the model since the beginning of training (stored as floats
            to avoid overflow).
        log_history (`List[Dict[str, float]]`, *optional*):
            The list of logs done since the beginning of training.
        best_metric (`float`, *optional*):
            When tracking the best model, the value of the best metric encountered so far.
        best_model_checkpoint (`str`, *optional*):
            When tracking the best model, the value of the name of the checkpoint for the best model encountered so
            far.
        is_local_process_zero (`bool`, *optional*, defaults to `True`):
            Whether or not this process is the local (e.g., on one machine if training in a distributed fashion on
            several machines) main process.
        is_world_process_zero (`bool`, *optional*, defaults to `True`):
            Whether or not this process is the global main process (when training in a distributed fashion on several
            machines, this is only going to be `True` for one process).
        is_hyper_param_search (`bool`, *optional*, defaults to `False`):
            Whether we are in the process of a hyper parameter search using Trainer.hyperparameter_search. This will
            impact the way original_data will be logged in TensorBoard.
        stateful_callbacks (`List[StatefulTrainerCallback]`, *optional*):
            Callbacks attached to the `Trainer` that should have their states be saved or restored.
            Relevent callbacks should implement a `state` and `from_state` function.
    """
    # state of training
    state_train: StateEnum = field(default=StateEnum.CLOSED, init=False)
    state_epoch: StateEnum = field(default=StateEnum.CLOSED, init=False)
    state_step: StateEnum = field(default=StateEnum.CLOSED, init=False)

    # epoch and step
    epoch: Optional[float or int] = None
    global_step: int = 0
    max_steps: int = 0
    max_epochs: int = 0
    logging_steps: int = 500
    eval_steps: int = 500
    save_steps: int = 500

    # train_batch_size: int = None
    num_train_epochs: int = 0
    num_input_tokens_seen: int = 0
    total_flos: float = 0
    train_lr: float = 0
    log_history: List[Dict[str, float]] = None
    best_metric: Optional[float] = None
    best_model_checkpoint: Optional[str] = None
    is_local_process_zero: bool = True
    is_world_process_zero: bool = True
    is_hyper_param_search: bool = False
    trial_name: str = None
    trial_params: Dict[str, Union[str, float, int, bool]] = None
    stateful_callbacks: List["TrainerCallback"] = None

    #
    valid_interval: int = 10
    batch_size: int =None
    train_sample_num:int = None
    valid_sample_num:int = None
    flag_autocast:bool = False
    loss_func_desc:str = None

    # epoch
    # epoch_validating:bool = False
    epoch_property:EpochProperty = EpochProperty.NONE
    temp_epoch_avg_loss:float = None
    epoch_avg_loss:float = None
    epoch_metric_data:Dict = None
    epoch_cost_time_ms: float = 0

    # step
    step_avg_loss :float = 0
    step_count:float = 0
    step_cost_time_ms:float = 0
    skip_loss:bool = False
    skip_backward:bool = False

    #save
    best_train_loss:float=None
    best_valid_loss:float=None
    save_name:str = "ModelSaved"

    # evaluate
    eval_current_dataset:str = "none"
    eval_time_per_item_ms :float = 0.

    # test prediction interactive
    interrupt_metric:str = "loss"
    interrupt_threshold:float = 0.1

    # info
    info_saved_model:set = None

    def __post_init__(self):
        if self.log_history is None:
            self.log_history = []
        if self.stateful_callbacks is None:
            self.stateful_callbacks = {}
        elif isinstance(self.stateful_callbacks, dict):
            # We are loading the callbacks in from the state file, no need to process them
            pass
        else:
            # Saveable callbacks get stored as dict of kwargs
            stateful_callbacks = {}
            for callback in self.stateful_callbacks:
                if not isinstance(callback, (ExportableState)):
                    raise TypeError(
                        f"All callbacks passed to be saved must inherit `ExportableState`, but received {type(callback)}"
                    )
                name = callback.__class__.__name__
                if name in stateful_callbacks:
                    # We can have multiple versions of the same callback
                    # if so, we store them as a list of states to restore
                    if not isinstance(stateful_callbacks[name], list):
                        stateful_callbacks[name] = [stateful_callbacks[name]]
                    stateful_callbacks[name].append(callback.state())
                else:
                    stateful_callbacks[name] = callback.state()
            self.stateful_callbacks = stateful_callbacks

        self.info_saved_model = set()

    def save_to_json(self, json_path: str):
        """Save the content of this instance in JSON format inside `json_path`."""
        json_string = json.dumps(dataclasses.asdict(self), indent=2, sort_keys=True) + "\n"
        with open(json_path, "w", encoding="utf-8") as f:
            f.write(json_string)

    @classmethod
    def load_from_json(cls, json_path: str):
        """Create an instance from the content of `json_path`."""
        with open(json_path, "r", encoding="utf-8") as f:
            text = f.read()
        return cls(**json.loads(text))


class ExportableState:
    """
    A class for objects that include the ability to have its state
    be saved during `Trainer._save_checkpoint` and loaded back in during
    `Trainer._load_from_checkpoint`.

    These must implement a `state` function that gets called during the respective
    Trainer function call. It should only include parameters and attributes needed to
    recreate the state at a particular time, to avoid utilizing pickle/maintain standard
    file IO writing.

    Example:

    ```python
    class EarlyStoppingCallback(TrainerCallback, ExportableState):
        def __init__(self, early_stopping_patience: int = 1, early_stopping_threshold: Optional[float] = 0.0):
            self.early_stopping_patience = early_stopping_patience
            self.early_stopping_threshold = early_stopping_threshold
            # early_stopping_patience_counter denotes the number of times validation metrics failed to improve.
            self.early_stopping_patience_counter = 0

        def state(self) -> dict:
            return {
                "args": {
                    "early_stopping_patience": self.early_stopping_patience,
                    "early_stopping_threshold": self.early_stopping_threshold,
                },
                "attributes": {
                    "early_stopping_patience_counter": self.early_stopping_patience_counter,
                }
            }
    ```"""

    def state(self) -> dict:
        raise NotImplementedError("You must implement a `state` function to utilize this class.")

    @classmethod
    def from_state(cls, state):
        instance = cls(**state["args"])
        for k, v in state["attributes"].items():
            setattr(instance, k, v)
        return instance


@dataclass
class TrainerControl(ExportableState):
    """
    A class that handles the [`Trainer`] control flow. This class is used by the [`TrainerCallback`] to activate some
    switches in the training loop.

    Args:
        should_training_stop (`bool`, *optional*, defaults to `False`):
            Whether or not the training should be interrupted.

            If `True`, this variable will not be set back to `False`. The training will just stop.
        should_epoch_stop (`bool`, *optional*, defaults to `False`):
            Whether or not the current epoch should be interrupted.

            If `True`, this variable will be set back to `False` at the beginning of the next epoch.
        should_save (`bool`, *optional*, defaults to `False`):
            Whether or not the model should be saved at this step.

            If `True`, this variable will be set back to `False` at the beginning of the next step.
        should_evaluate (`bool`, *optional*, defaults to `False`):
            Whether or not the model should be evaluated at this step.

            If `True`, this variable will be set back to `False` at the beginning of the next step.
        should_log (`bool`, *optional*, defaults to `False`):
            Whether or not the logs should be reported at this step.

            If `True`, this variable will be set back to `False` at the beginning of the next step.
    """
    reporter : SubReporter = field(default = None, init=True, repr=False)
    should_training_stop: bool = False
    should_epoch_stop: bool = False
    should_valid:bool = False
    should_calculate_metric:bool = False
    should_save: bool = False
    should_evaluate: bool = False
    should_log: bool = False
    flag_step_nan: bool = False
    # writer: SummaryWriter = None
    # def __post_init__(self):
    #     if self.reporter is not None:
    #         self.writer:SummaryWriter = self.reporter.tb_summary_writer
    proxy_cmd_handle:list[Callable[[str],bool]] = None
    def __post_init__(self):
        self.proxy_cmd_handle = []

    def proxy_cmd_handle_register(self,cmd_handle):
        self.proxy_cmd_handle.append(cmd_handle)

    @property
    def writer(self)->SummaryWriter:
        return self.reporter.terminal_ctrl.get_tb_summary_writer()

    def _new_training(self):
        """Internal method that resets the variable for a new training."""
        self.should_training_stop = False

    def _new_epoch(self):
        """Internal method that resets the variable for a new epoch."""
        self.should_epoch_stop = False
        # self.should_epoch_stop = False

    def _new_step(self):
        """Internal method that resets the variable for a new step."""
        self.should_save = False
        self.should_evaluate = False
        self.should_log = False

    def command_process(self):
        if self.reporter is None:
            return
        cmd = self.reporter.terminal_ctrl.get_one_cmd()
        self.drive_command(cmd, silence=False)

    def drive_command(self, cmd, silence=True):
        if cmd is None:
            return
        elif cmd == "stop":
            self.should_training_stop = True
            self.reporter(f"[!] command {cmd}: stop the train after the epoch", flag_stdout= not silence, flag_log=True)
            return
        elif cmd == "shutdown":
            self.should_epoch_stop = True
            self.should_training_stop = True
            self.reporter(f"[!] command {cmd}: shutdown the train now", flag_stdout= not silence, flag_log=True)
            return
        else:
            for handle in self.proxy_cmd_handle:
                result = handle(cmd)
                if result:
                    return
            self.reporter(f"[!] unknown command {cmd}", flag_stdout= not silence, flag_log=True)
            return

    def command_dict(self):
        return {"stop":"stop the epoch after the epoch",
                "shutdown":"shutdown the train now",}
    def state(self) -> dict:
        return {
            "args": {
                "should_training_stop": self.should_training_stop,
                "should_epoch_stop": self.should_epoch_stop,
                "should_save": self.should_save,
                "should_evaluate": self.should_evaluate,
                "should_log": self.should_log,
            },
            "attributes": {},
        }


class TrainerCallback:
    # no-format
    """
    A class for objects that will inspect the state of the training loop at some events and take some decisions. At
    each of those events the following arguments are available:

    Args:
        args ([`TrainingArguments`]):
            The training arguments used to instantiate the [`Trainer`].
        state ([`TrainerState`]):
            The current state of the [`Trainer`].
        control ([`TrainerControl`]):
            The object that is returned to the [`Trainer`] and can be used to make some decisions.
        model ([`PreTrainedModel`] or `torch.nn.Module`):
            The model being trained.
        tokenizer ([`PreTrainedTokenizer`]):
            The tokenizer used for encoding the original_data. This is deprecated in favour of `processing_class`.
        processing_class ([`PreTrainedTokenizer` or `BaseImageProcessor` or `ProcessorMixin` or `FeatureExtractionMixin`]):
            The processing class used for encoding the original_data. Can be a tokenizer, a processor, an image processor or a feature extractor.
        optimizer (`torch.optim.Optimizer`):
            The optimizer used for the training steps.
        lr_scheduler (`torch.optim.lr_scheduler.LambdaLR`):
            The scheduler used for setting the learning rate.
        train_dataloader (`torch.utils.original_data.DataLoader`, *optional*):
            The current dataloader used for training.
        eval_dataloader (`torch.utils.original_data.DataLoader`, *optional*):
            The current dataloader used for evaluation.
        metrics (`Dict[str, float]`):
            The metrics computed by the last evaluation phase.

            Those are only accessible in the event `on_evaluate`.
        logs  (`Dict[str, float]`):
            The values to log.

            Those are only accessible in the event `on_log`.

    The `control` object is the only one that can be changed by the callback, in which case the event that changes it
    should return the modified version.

    The argument `args`, `state` and `control` are positionals for all events, all the others are grouped in `kwargs`.
    You can unpack the ones you need in the signature of the event using them. As an example, see the code of the
    simple [`~transformers.PrinterCallback`].

    Example:

    ```python
    class PrinterCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            _ = logs.pop("total_flos", None)
            if state.is_local_process_zero:
                print(logs)
    ```"""

    def on_init_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event called at the end of the initialization of the [`Trainer`].
        """
        pass

    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event called at the beginning of training.
        """
        pass

    def on_train_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event called at the end of training.
        """
        pass

    def on_predict_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event called at the beginning of prediction.
        """
        pass

    def on_predict_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event called at the end of prediction.
        """
        pass

    def on_epoch_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event called at the beginning of an epoch.
        """
        pass

    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event called at the end of an epoch.
        """
        pass

    def on_step_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event called at the beginning of a training step. If using gradient accumulation, one training step might take
        several inputs.
        """
        pass

    def on_pre_optimizer_step(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event called before the optimizer step but after gradient clipping. Useful for monitoring gradients.
        """
        pass

    def on_optimizer_step(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event called after the optimizer step but before gradients are zeroed out. Useful for monitoring gradients.
        """
        pass

    def on_substep_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event called at the end of an substep during gradient accumulation.
        """
        pass

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event called at the end of a training step. If using gradient accumulation, one training step might take
        several inputs.
        """
        pass

    def on_evaluate(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event called after an evaluation phase.
        """
        pass

    def on_predict(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, metrics, **kwargs):
        """
        Event called after a successful prediction.
        """
        pass



    def on_prediction_step(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event called after a prediction step.
        """
        pass

    def on_save(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event called after a checkpoint save.
        """
        pass

    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event called after logging the last logs.
        """
        pass

    def on_nan_happen(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event called when a NaN is happened.
        """
        pass


class CallbackHandler(TrainerCallback):
    """Internal class that just calls the list of callbacks in order."""

    def __init__(self, callbacks, model, processing_class, optimizer, lr_scheduler):
        self.callbacks = []
        for cb in callbacks:
            self.add_callback(cb)
        # self.reporter:SubReporter = reporter
        self.model = model
        self.processing_class = processing_class
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.train_dataloader = None
        self.eval_dataloader = None

        if not any(isinstance(cb, DefaultFlowCallback) for cb in self.callbacks):
            warnings.warn(
                "The Trainer will not work properly if you don't have a `DefaultFlowCallback` in its callbacks. You\n"
                + "should add one before training with `trainer.add_callback(DefaultFlowCallback). The current list of"
                + "callbacks is\n:"
                + self.callback_list
            )

    def add_callback(self, callback):
        cb = callback() if isinstance(callback, type) else callback
        cb_class = callback if isinstance(callback, type) else callback.__class__
        if cb_class in [c.__class__ for c in self.callbacks]:
            warnings.warn(
                f"You are adding a {cb_class} to the callbacks of this Trainer, but there is already one. The current"
                + "list of callbacks is\n:"
                + self.callback_list
            )
        self.callbacks.append(cb)

    def pop_callback(self, callback):
        if isinstance(callback, type):
            for cb in self.callbacks:
                if isinstance(cb, callback):
                    self.callbacks.remove(cb)
                    return cb
        else:
            for cb in self.callbacks:
                if cb == callback:
                    self.callbacks.remove(cb)
                    return cb

    def remove_callback(self, callback):
        if isinstance(callback, type):
            for cb in self.callbacks:
                if isinstance(cb, callback):
                    self.callbacks.remove(cb)
                    return
        else:
            self.callbacks.remove(callback)

    @property
    def callback_list(self):
        return "\n".join(cb.__class__.__name__ for cb in self.callbacks)

    def on_init_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl):
        return self.call_event("on_init_end", args, state, control)

    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl):
        control.should_training_stop = False
        return self.call_event("on_train_begin", args, state, control)

    def on_train_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl):
        return self.call_event("on_train_end", args, state, control)

    def on_predict_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl):
        return self.call_event("on_predict_begin", args, state, control)

    def on_predict_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl):
        return self.call_event("on_predict_end", args, state, control)

    def on_epoch_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl):
        control.should_epoch_stop = False
        return self.call_event("on_epoch_begin", args, state, control)

    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl):
        return self.call_event("on_epoch_end", args, state, control)

    def on_step_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl):
        control.should_log = False
        control.should_evaluate = False
        control.should_save = False
        return self.call_event("on_step_begin", args, state, control)

    def on_pre_optimizer_step(self, args: TrainingArguments, state: TrainerState, control: TrainerControl):
        return self.call_event("on_pre_optimizer_step", args, state, control)

    def on_optimizer_step(self, args: TrainingArguments, state: TrainerState, control: TrainerControl):
        return self.call_event("on_optimizer_step", args, state, control)

    def on_substep_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl):
        return self.call_event("on_substep_end", args, state, control)

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl):
        return self.call_event("on_step_end", args, state, control)

    def on_evaluate(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, metrics):
        control.should_evaluate = False
        return self.call_event("on_evaluate", args, state, control, metrics=metrics)

    def on_predict(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, metrics):
        return self.call_event("on_predict", args, state, control, metrics=metrics)

    def on_save(self, args: TrainingArguments, state: TrainerState, control: TrainerControl):
        control.should_save = False
        return self.call_event("on_save", args, state, control)

    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, logs):
        control.should_log = False
        return self.call_event("on_log", args, state, control, logs=logs)

    def on_prediction_step(self, args: TrainingArguments, state: TrainerState, control: TrainerControl):
        return self.call_event("on_prediction_step", args, state, control)

    def on_nan_happen(self, args: TrainingArguments, state: TrainerState, control: TrainerControl):
        """
        Event called when a NaN is happened.
        """
        return self.call_event("on_nan_happen", args, state, control)

    def call_event(self, event, args, state, control, **kwargs):
        for callback in self.callbacks:
            result = getattr(callback, event)(
                args,
                state,
                control,
                model=self.model,
                processing_class=self.processing_class,
                optimizer=self.optimizer,
                lr_scheduler=self.lr_scheduler,
                train_dataloader=self.train_dataloader,
                eval_dataloader=self.eval_dataloader,
                **kwargs,
            )
            # A Callback can skip the return of `control` if it doesn't change it.
            if result is not None:
                control = result
        return control


class DefaultFlowCallback(TrainerCallback):
    """
    A [`TrainerCallback`] that handles the default flow of the training loop for logs, evaluation and checkpoints.
    """

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        # Log
        # if state.global_step == 1 and args.logging_first_step:
        #     control.should_log = True
        # if args.logging_strategy == IntervalStrategy.STEPS and state.global_step % state.logging_steps == 0:
        #     control.should_log = True
        #
        # # Evaluate
        # if (
        #     args.eval_strategy == IntervalStrategy.STEPS
        #     and state.global_step % state.eval_steps == 0
        #     and args.eval_delay <= state.global_step
        # ):
        #     control.should_evaluate = True

        # Save
        # if (
        #     args.save_strategy == SaveStrategy.STEPS
        #     and state.save_steps > 0
        #     and state.global_step % state.save_steps == 0
        # ):
        #     control.should_save = True

        # End training
        # if state.global_step >= state.max_steps > 0:
        #     control.should_training_stop = True
        #     # Save the model at the end if we have a save strategy
        #     if args.save_strategy not in [SaveStrategy.NO, SaveStrategy.BEST]:
        #         control.should_save = True

        return control

    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        # Log
        if args.logging_strategy == IntervalStrategy.EPOCH:
            control.should_log = True

        # Evaluate
        # if args.eval_strategy == IntervalStrategy.EPOCH and args.eval_delay <= state.epoch:
        #     control.should_evaluate = True

        # # Save
        # if args.save_strategy == SaveStrategy.EPOCH:
        #     control.should_save = True

        return control


class ProgressCallback(TrainerCallback):
    """
    A [`TrainerCallback`] that displays the progress of training or evaluation.
    You can modify `max_str_len` to control how long strings are truncated when logging.
    """

    def __init__(self, max_str_len: int = 100):
        """
        Initialize the callback with optional max_str_len parameter to control string truncation length.

        Args:
            max_str_len (`int`):
                Maximum length of strings to display in logs.
                Longer strings will be truncated with a message.
        """
        self.training_bar = None
        self.prediction_bar = None
        self.max_str_len = max_str_len

    def on_train_begin(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            self.training_bar = tqdm(total=state.max_steps, dynamic_ncols=True)
        self.current_step = 0

    def on_step_end(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            self.training_bar.update(state.global_step - self.current_step)
            self.current_step = state.global_step

    def on_prediction_step(self, args, state, control, eval_dataloader=None, **kwargs):
        if state.is_world_process_zero and has_length(eval_dataloader):
            if self.prediction_bar is None:
                self.prediction_bar = tqdm(
                    total=len(eval_dataloader), leave=self.training_bar is None, dynamic_ncols=True
                )
            self.prediction_bar.update(1)

    def on_evaluate(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            if self.prediction_bar is not None:
                self.prediction_bar.close()
            self.prediction_bar = None

    def on_predict(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            if self.prediction_bar is not None:
                self.prediction_bar.close()
            self.prediction_bar = None

    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.is_world_process_zero and self.training_bar is not None:
            # make a shallow copy of logs so we can mutate the fields copied
            # but avoid doing any value pickling.
            shallow_logs = {}
            for k, v in logs.items():
                if isinstance(v, str) and len(v) > self.max_str_len:
                    shallow_logs[k] = (
                        f"[String too long to display, length: {len(v)} > {self.max_str_len}. "
                        "Consider increasing `max_str_len` if needed.]"
                    )
                else:
                    shallow_logs[k] = v
            _ = shallow_logs.pop("total_flos", None)
            # round numbers so that it looks better in console
            if "epoch" in shallow_logs:
                shallow_logs["epoch"] = round(shallow_logs["epoch"], 2)
            self.training_bar.write(str(shallow_logs))

    def on_train_end(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            self.training_bar.close()
            self.training_bar = None


class PrinterCallback(TrainerCallback):
    """
    A bare [`TrainerCallback`] that just prints the logs.
    """

    def on_log(self, args, state, control, logs=None, **kwargs):
        _ = logs.pop("total_flos", None)
        if state.is_local_process_zero:
            print(logs)


class EarlyStoppingCallback(TrainerCallback, ExportableState):
    """
    A [`TrainerCallback`] that handles early stopping.

    Args:
        early_stopping_patience (`int`):
            Use with `metric_for_best_model` to stop training when the specified metric worsens for
            `early_stopping_patience` evaluation calls.
        early_stopping_threshold(`float`, *optional*):
            Use with TrainingArguments `metric_for_best_model` and `early_stopping_patience` to denote how much the
            specified metric must improve to satisfy early stopping conditions. `

    This callback depends on [`TrainingArguments`] argument *load_best_model_at_end* functionality to set best_metric
    in [`TrainerState`]. Note that if the [`TrainingArguments`] argument *save_steps* differs from *eval_steps*, the
    early stopping will not occur until the next save step.
    """

    def __init__(self, early_stopping_patience: int = 1, early_stopping_threshold: Optional[float] = 0.0):
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_threshold = early_stopping_threshold
        # early_stopping_patience_counter denotes the number of times validation metrics failed to improve.
        self.early_stopping_patience_counter = 0

    def check_metric_value(self, args, state, control, metric_value):
        # best_metric is set by code for load_best_model
        operator = np.greater if args.greater_is_better else np.less
        if state.best_metric is None or (
            operator(metric_value, state.best_metric)
            and abs(metric_value - state.best_metric) > self.early_stopping_threshold
        ):
            self.early_stopping_patience_counter = 0
        else:
            self.early_stopping_patience_counter += 1

    def on_train_begin(self, args, state, control, **kwargs):
        assert args.load_best_model_at_end, "EarlyStoppingCallback requires load_best_model_at_end = True"
        assert (
            args.metric_for_best_model is not None
        ), "EarlyStoppingCallback requires metric_for_best_model is defined"
        assert (
            args.eval_strategy != IntervalStrategy.NO
        ), "EarlyStoppingCallback requires IntervalStrategy of steps or epoch"

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        metric_to_check = args.metric_for_best_model
        if not metric_to_check.startswith("eval_"):
            metric_to_check = f"eval_{metric_to_check}"
        metric_value = metrics.get(metric_to_check)

        if metric_value is None:
            warnings.warn(
                f"early stopping required metric_for_best_model, but did not find {metric_to_check} so early stopping"
                " is disabled"
            )
            return

        self.check_metric_value(args, state, control, metric_value)
        if self.early_stopping_patience_counter >= self.early_stopping_patience:
            control.should_training_stop = True
            control.reporter("Stopped Training Cause Early Stopping detector")

    def state(self) -> dict:
        return {
            "args": {
                "early_stopping_patience": self.early_stopping_patience,
                "early_stopping_threshold": self.early_stopping_threshold,
            },
            "attributes": {
                "early_stopping_patience_counter": self.early_stopping_patience_counter,
            },
        }

class CustomCallback(TrainerCallback):

    def on_init_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event called at the end of the initialization of the [`Trainer`].
        """
        control.reporter("[!] Trainer init finished")
        return control

    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event called at the beginning of training.
        """
        control.should_training_stop = False
        control.reporter("***** Running training *****").log().md()
        control.reporter(f" Num Epochs = {state.max_epochs}").log().md()
        control.reporter(f" Valid Interval = {state.valid_interval}").log().md()
        control.reporter(f" Batch Size = {state.batch_size}").log().md()
        control.reporter(f" Train Samples = {state.train_sample_num}").log().md()
        control.reporter(f" Valid Samples = {state.valid_sample_num}").log().md()
        control.reporter(f" Auto Cast = {state.flag_autocast}").log().md()
        control.reporter(f" Loss Func = {state.loss_func_desc}").log().md()
        control.reporter(f" Init Lr = {state.train_lr}").log().md()

        torch_seed = torch.random.initial_seed()
        random_state = np.random.get_state()
        numpy_seed = random_state[1][0]  # 种子通常在状态数组的第二个元素中
        control.reporter(f"* PyTorch random seed: {torch_seed}",flag_stdout=True, flag_md=True, flag_log=True)
        control.reporter(f"* NumPy random seed: {numpy_seed}", flag_stdout=True, flag_md=True, flag_log=True)
        control.reporter(f"* Args seed: {args.seed}", flag_stdout=True, flag_md=True, flag_log=True)
        if state.epoch_property != EpochProperty.TEST:
            if control.writer is not None:
                control.reporter(f" TensorboardPath = {args.reporter.terminal_ctrl.tensorboard_path}").log().md()
        control.reporter("****************************").log().md()
        state.epoch_property = EpochProperty.TRAIN

        # Save Strategy
        if args.save_strategy == SaveStrategy.CUSTOM:
            if args.save_init:
                control.should_save = True
                state.save_name="init_{}".format(state.epoch_property.value)
                # state.save_name="init_{}".format("valid" if state.epoch_property==EpochProperty.VALID else "train")

        return control

    def on_train_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event called at the end of training.
        """
        control.reporter("[!] Train end!").log().md()
        control.reporter(f"log save dir:{os.path.relpath(args.reporter_save_dir, args.project_root)}").log().md()
        if state.epoch_property != EpochProperty.TEST:
            if control.writer is not None:
                control.reporter(f" tensorboard Dir = {args.reporter.terminal_ctrl.tensorboard_path}").log().md()

        args.done_train = True
        reporter_save_dir = args.save_dir_prefix
        if "trained" not in reporter_save_dir.split("_"):
            reporter_save_dir += "_trained"
            args.rename_reporter_folder_name(reporter_save_dir)
        # Save Strategy
        if args.save_strategy == SaveStrategy.CUSTOM:
            if args.save_last:
                control.should_save = True
                state.save_name="last_{}".format(state.epoch_property.value)

        self.on_generate_end(args, state, control)
        return control

    def on_epoch_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event called at the beginning of an epoch.
        """
        control.should_epoch_stop = False
        control.should_calculate_metric = False
        state.epoch_metric_data = None
        # whether calculate loss and backward
        if state.epoch_property == EpochProperty.TRAIN:
            state.skip_loss = False
            state.skip_backward = False
        elif state.epoch_property == EpochProperty.VALID:
            state.skip_loss = False
            state.skip_backward = True
        elif state.epoch_property == EpochProperty.TEST:
            state.skip_loss = True
            state.skip_backward = True

        # whether it need validation
        if state.epoch_property == EpochProperty.TRAIN:
            if (state.epoch + 1) % args.valid_interval == 0:
                control.should_valid = True
        elif state.epoch_property == EpochProperty.VALID:
            control.should_valid = False

        if args.metric_strategy == MetricStrategy.EPOCH:
            control.should_calculate_metric = True
        elif args.metric_strategy == MetricStrategy.VALID and state.epoch_property == EpochProperty.VALID:
            control.should_calculate_metric = True
        elif args.metric_strategy == MetricStrategy.TRAIN and state.epoch_property == EpochProperty.TRAIN:
            control.should_calculate_metric = True
        elif state.epoch_property == EpochProperty.TEST:
            control.should_calculate_metric = True

        control.reporter("**********************************************************")
        state.epoch_cost_time_ms = 0
        return control

    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event called at the end of an epoch.
        """
        control.reporter(f"\tepoch = {state.epoch+1}").log()
        control.reporter(f"\taverage loss = {state.epoch_avg_loss}").log()
        control.reporter(f"\tlearning rate = {state.train_lr}").log()

        # loss in csv
        now_time = get_str_time(Date=False, dateDiv="/", timeDiv=":", datetimeDiv=" ")
        epoch_time_elapsed = state.epoch_cost_time_ms / 1e3
        if state.epoch_property == EpochProperty.TRAIN:
            # 如果当前是训练阶段
            if control.should_valid:
                # 该epoch需要测试
                state.temp_epoch_avg_loss = state.epoch_avg_loss
            else:
                # 该epoch不需要测试
                control.reporter.add_line_csv([state.epoch_property.value,
                                               now_time,
                                               state.epoch + 1,
                                               state.epoch_avg_loss,
                                               None,
                                               state.train_lr,
                                               epoch_time_elapsed
                                               ])
        elif state.epoch_property == EpochProperty.VALID:
            # 当前是测试阶段
            control.reporter.add_line_csv([state.epoch_property.value,
                                           now_time,
                                           state.epoch+1,
                                           state.temp_epoch_avg_loss,
                                           state.epoch_avg_loss,
                                           state.train_lr,
                                           epoch_time_elapsed
                                           ])

        def check_tensorboard():
            if state.epoch_property != EpochProperty.TEST:
                if control.writer is not None:
                    return True
            return False

        # Save Strategy

        if args.save_strategy is not SaveStrategy.NO and not control.flag_step_nan:
            epoch_info = "" if args.save_covered else f"_Ep{state.epoch+1}"
            epoch_property = state.epoch_property.value
            save_best_valid = args.save_strategy == SaveStrategy.BEST or args.save_strategy == SaveStrategy.BEST_VALID
            save_best_train = args.save_strategy == SaveStrategy.BEST or args.save_strategy == SaveStrategy.BEST_TRAIN
            if args.save_strategy == SaveStrategy.CUSTOM:
                save_best_train = save_best_valid = args.save_best
                if state.epoch + 1 in args.save_epoch_ids and state.epoch_property==EpochProperty.TRAIN:
                    control.should_save = True
                    state.save_name = "cp_{}{}".format(epoch_property, state.epoch + 1)
            elif args.save_strategy == SaveStrategy.EPOCH:
                control.should_save=True
                state.save_name="last_{}{}".format(epoch_property, epoch_info)
            # if args.save_strategy == SaveStrategy.BEST:
            if state.epoch_property == EpochProperty.VALID:
                if state.best_valid_loss is None or state.epoch_avg_loss < state.best_valid_loss:
                    state.best_valid_loss = state.epoch_avg_loss
                    if save_best_valid and not control.should_save:
                        control.should_save=True
                        state.save_name = "best_{}{}".format("valid", epoch_info)
            elif state.epoch_property==EpochProperty.TRAIN:
                if state.best_train_loss is None or state.epoch_avg_loss < state.best_train_loss:
                    state.best_train_loss = state.epoch_avg_loss
                    if save_best_train and not control.should_save:
                        control.should_save=True
                        state.save_name = "best_{}{}".format("train", epoch_info)

        # tensorboard
        if check_tensorboard():
            if state.epoch_property == EpochProperty.TRAIN:
                control.writer.add_scalar("loss/train", state.epoch_avg_loss, state.epoch)
                control.writer.add_scalar("learning_rate/train", state.train_lr, state.epoch)
            elif state.epoch_property == EpochProperty.VALID:
                control.writer.add_scalar("loss/valid", state.epoch_avg_loss, state.epoch)

        # calculate metric
        if state.epoch_metric_data is not None:
            control.reporter(f"metrics:").log()
            for k, v in state.epoch_metric_data.items():

                if re.match("detail_", k):
                    pass
                    # control.reporter(f"--{k}:\n {v}",flag_log=True)
                else:
                    control.reporter(f"--{k}:\n {v}").log()
                    if check_tensorboard():
                        if isinstance(v, float):
                            control.writer.add_scalar(f"{state.epoch_property.value}", v, state.epoch)
                        if isinstance(v, pd.DataFrame):
                            for i in range(len(v)):
                                tmp = v.iloc[i]
                                for j in range(len(tmp)):
                                    name = tmp.name
                                    idx = tmp.index[j]
                                    num = tmp.iloc[j]
                                    control.writer.add_scalar(f"{name}/{idx}/{state.epoch_property.value}",
                                                              num, state.epoch)
                    # control.reporter(f"--{k}:\n {v}").log().md()

        return control



    def on_step_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event called at the beginning of a training step. If using gradient accumulation, one training step might take
        several inputs.
        """
        # check if there is input
        control.reporter.check_input()
        # control.reporter("[!] Train end!\n")
        return control

    def on_pre_optimizer_step(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event called before the optimizer step but after gradient clipping. Useful for monitoring gradients.
        """
        control.reporter("[!] pre_optimizer_step!\n")
        return control

    def on_optimizer_step(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event called after the optimizer step but before gradients are zeroed out. Useful for monitoring gradients.
        """
        control.reporter("[!] optimizer_step!\n")
        return control

    def on_substep_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event called at the end of an substep during gradient accumulation.
        """
        control.reporter("[!] sub step end!\n")
        return control

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event called at the end of a training step. If using gradient accumulation, one training step might take
        several inputs.
        """
        # control.reporter("[!] step end!\n")
        control.command_process()
        state.step_count += 1
        state.epoch_cost_time_ms += state.step_cost_time_ms
        return control

    def on_evaluate(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event called after an evaluation phase.
        """
        control.reporter("[!] dataset <{}> evaluate finish".format(state.eval_current_dataset))
        control.reporter("    eval_time:{} ms".format(state.eval_time_per_item_ms))
        return control

    # def on_predict(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, metrics, **kwargs):
    #     """
    #     Event called after a successful prediction.
    #     """
    #     control.reporter("****************************************************")
    #     # control.reporter("[!] Predict finish\n")
    #
    #     control.reporter(f"[+] log in path(abs):{args.reporter_save_dir}")
    #     args.done_predict = True
    #     return control

    def on_predict_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        return control

    def on_predict_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event called after a successful prediction.
        """
        control.reporter("****************************************************")
        # control.reporter("[!] Predict finish\n")

        control.reporter(f"[+] log in path(abs):{args.reporter_save_dir}")
        self.on_generate_end(args, state, control)
        # args.done_predict = True
        # reporter_save_dir = args.save_dir_prefix
        # if "predicted" not in reporter_save_dir.split("_"):
        #     reporter_save_dir += "_predicted"
        #     args.rename_reporter_folder_name(reporter_save_dir)
        return control


    def on_save(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event called after a checkpoint save.
        """
        control.should_save = False
        state.info_saved_model.add(state.save_name)
        control.reporter("[!] Saved model in <{}>!".format(os.path.relpath(os.path.join(args.model_save_dir, state.save_name), args.project_root)))
        return control

    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event called after logging the last logs.
        """
        # control.reporter("[!] logged!\n")
        return control

    def on_prediction_step(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event called after a prediction step.
        """
        return control

    def on_generate_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if len(state.info_saved_model) != 0:
            control.reporter("[!] saved model list:").log().md()
            control.reporter("\t dir:{}".format(os.path.relpath(os.path.join(args.model_save_dir), args.project_root)))
            for model_name in state.info_saved_model:
                control.reporter(f"\t   - {model_name}").log().md()
            return control

    def on_nan_happen(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event called when a NaN is happened.
        """
        control.reporter(f"\n [!-] There is a nan appears").log()
        control.should_training_stop = True
        control.should_epoch_stop = True
        control.should_valid = False
        return control

class PredictionInteraction(TrainerCallback):
    def __init__(self,interrupt_threshold=0.2):
        super().__init__()
        self.interrupt_threshold = interrupt_threshold

    """
    只是为了交互性而存在的模块
    """
    def on_init_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        state.interrupt_threshold = self.interrupt_threshold
        control.reporter(f"[+] Prediction Interaction mode on! Interrupt Threshold is '{state.interrupt_metric}' > {state.interrupt_threshold}").md().log()
        control.reporter("[!] Note: This mode forces: test_batch == 1; calculate loss and metric by step; \n"
                         "[-] May result in performance degradation.").log()
        args.batch_size = 1
        def set_threshold(cmd):
            try:
                n = float(cmd)
                state.interrupt_threshold = n
                control.reporter(f"[!] change the threshold to {n}")
                return True
            except ValueError:
                return False
        control.proxy_cmd_handle_register(set_threshold)
        return control

    def on_predict_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if state.epoch_property == EpochProperty.TEST:
            control.reporter(f"[+] Interrupt Threshold is '{state.interrupt_metric}' > {state.interrupt_threshold}")
        return control

    def on_predict_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        return control

    def on_prediction_step(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        return control

    def on_step_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if state.epoch_property == EpochProperty.TEST:
            state.skip_loss = False

        return control

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if state.epoch_property == EpochProperty.TEST:
            if state.step_avg_loss > state.interrupt_threshold:
                control.reporter(f"\n[!-] step[{state.step_count}] {state.interrupt_metric} == {state.step_avg_loss} > {state.interrupt_threshold}").log()
                control.reporter(f"\t Please input your command [shutdown or float for new threshold]:")
                cmd = input("\t ")
                if cmd == 'shutdown':
                    # control.reporter("Terminate Reasoning").log()
                    control.drive_command("shutdown")
                    control.reporter(f"[!] shutdown the prediction now")
                else:
                    try:
                        n = float(cmd)
                        state.interrupt_threshold = n
                        control.reporter(f"[!] change the threshold to {n}")
                    except ValueError:
                        control.reporter(f"[-] invalid command: [{cmd}]")

        return control