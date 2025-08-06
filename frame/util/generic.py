import warnings
from collections import OrderedDict, UserDict
from dataclasses import fields, is_dataclass
# from typing import Any, ContextManager, Dict, Iterable, List, Optional, Tuple, TypedDict
from typing import Any, Dict, List, NamedTuple, Optional, Tuple, Union
import numpy as np
import torch

class ModelOutput(OrderedDict):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Subclasses of ModelOutput must use the @dataclass decorator
        # This check is done in __init__ because the @dataclass decorator operates after __init_subclass__
        # issubclass() would return True for issubclass(ModelOutput, ModelOutput) when False is needed
        # Just need to check that the current class is not ModelOutput
        # ! : all the value should be Tensor or Array
        # ! : It is recommended to define its subclasses inside Networks ( as an attr)
        is_modeloutput_subclass = self.__class__ != ModelOutput

        if is_modeloutput_subclass and not is_dataclass(self):
            raise TypeError(
                f"{self.__module__}.{self.__class__.__name__} is not a dataclasss."
                " This is a subclass of ModelOutput and so must use the @dataclass decorator."
            )


    def __delitem__(self, *args, **kwargs):
        raise Exception(f"You cannot use ``__delitem__`` on a {self.__class__.__name__} instance.")

    def setdefault(self, *args, **kwargs):
        raise Exception(f"You cannot use ``setdefault`` on a {self.__class__.__name__} instance.")

    def pop(self, *args, **kwargs):
        raise Exception(f"You cannot use ``pop`` on a {self.__class__.__name__} instance.")

    def update(self, *args, **kwargs):
        raise Exception(f"You cannot use ``update`` on a {self.__class__.__name__} instance.")

    def __getitem__(self, k):
        if isinstance(k, str):
            inner_dict = dict(self.items())
            return inner_dict[k]
        else:
            return self.to_tuple()[k]

    def __setattr__(self, name, value):
        # if name in self.keys() and value is not None:
        #     # Don't call self.__setitem__ to avoid recursion errors
        #     super().__setitem__(name, value)
        super().__setitem__(name, value)
        super().__setattr__(name, value)

    def __setitem__(self, key, value):
        # Will raise a KeyException if needed
        super().__setitem__(key, value)
        # Don't call self.__setattr__ to avoid recursion errors
        super().__setattr__(key, value)

    def __reduce__(self):
        if not is_dataclass(self):
            return super().__reduce__()
        callable, _args, *remaining = super().__reduce__()
        args = tuple(getattr(self, field.name) for field in fields(self))
        return callable, args, *remaining

    def to_tuple(self) -> Tuple[Any]:
        """
        Convert self to a tuple containing all the attributes/keys that are not `None`.
        """
        return tuple(self[k] for k in self.keys())

class EvalPrediction(dict):
    def __init__(self, output, target, sample = None, input_args=None, loss = None,
                 concatenate=True):
        """
        :param concatenate: 为了适应batch like的离散数据，提供选项使输入进行一个合并
        :param input_args:        args需要与prediction等长度一一对应
        """
        self.length=0

        self.concatenate = concatenate
        self.elements = tuple()
        self.output = self.element_add(output)
        self.target = self.element_add(target)
        self.sample = self.element_add(sample)
        self.loss = self.element_add(loss)

        # self.args = args
        input_args = [] if input_args is None else input_args
        self._input_args = [np.array(sum(lst, ())) for lst in list(zip(*input_args))]
        for i, arg in enumerate(self._input_args):
            self.__setattr__(f"arg_{i}",arg)
        self.elements += (*self._input_args,)
        # if args is not None:
        #     self.args = [sum(lst,()) for lst in list(zip(*args))]
        #     self.elements += (*self.args,)
        # else:
        #     self.args = []
        self.build()
        super().__init__({"sample": self.sample, "output": self.output, "target": self.target, "loss": self.loss,
                          **{f"arg_{i}":sth for i,sth in enumerate(self._input_args)},
                          } )

    def element_add(self, data):
        if data is None:
            return None
        if self.concatenate:
            data = np.concatenate(data)
        self.elements += (data,)
        return data

    def build(self):
        length = [len(data) for data in self.elements]
        if len(set(length))==1:
            self.length = length[0]
        else:
            self.length = 0

    def set_item(self,key,data,concatenate=True):
        if concatenate:
            data = np.concatenate(data)
        # self.__setattr__(key, data)
        self[key] = data
        self.__setattr__(key, data)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if isinstance(idx, int):
            if idx < 0 or idx >= len(self):
                raise IndexError("tuple index out of range")
            return [element[idx] for element in self.elements]
        else:
            return super().__getitem__(idx)

class EpochSummary(EvalPrediction):
    def __init__(self, detach_model_output: ModelOutput, sample, target, args):
        output = detach_model_output["output"]
        loss = detach_model_output["loss"]
        super().__init__(output, target, sample, args,loss, True)
        for k,v in detach_model_output.items():
            if k not in ["output", "loss"]:
                self.__setattr__(k, np.concatenate(v))
                self[k] = np.concatenate(v)

# def __init__(self, output, target, sample = None, *args, loss = None, concatenate=True, **kwargs):
#     super().__init__(output, target, sample, args, loss, concatenate)

#
#
#
#
# class EvalLoopOutput(NamedTuple):
#     predictions: Union[np.ndarray, Tuple[np.ndarray]]
#     label_ids: Optional[Union[np.ndarray, Tuple[np.ndarray]]]
#     metrics: Optional[Dict[str, float]]
#     num_samples: Optional[int]
#
#
# class PredictionOutput(NamedTuple):
#     predictions: Union[np.ndarray, Tuple[np.ndarray]]
#     label_ids: Optional[Union[np.ndarray, Tuple[np.ndarray]]]
#     metrics: Optional[Dict[str, float]]
#
#
# class TrainOutput(NamedTuple):
#     global_step: int
#     training_loss: float
#     metrics: Dict[str, float]

class MetricCalculate:
    def __init__(self):
        self.metrics = {}

    def add_metric(self, name, metric):
        if name not in self.metrics:
            self.metrics[name] = metric
        else:
            warnings.warn("There is already a metric method [{}] in Class, Override.".format(name))
            self.metrics[name] = metric
        return self

    def __call__(self,eval_predicts, *args, **kwargs)-> Dict[str, float]:
        rtn = dict()
        for name, metric in self.metrics.items():
            rtn[name] = metric(*eval_predicts, *args, **kwargs)
        return rtn

    def __repr__(self):
        rtn = "MetricCalculate:\n"
        for name, metric in self.metrics.items():
            rtn += f"\t{name}: {metric}\n"

    @property
    def names(self):
        return self.metrics.keys()



