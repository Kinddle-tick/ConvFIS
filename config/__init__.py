import yaml
import os
import torch
from functools import lru_cache
from typing import FrozenSet
import configparser
from .import_utils import (
    get_platform,
    import_tqdm,
    is_jupyter,
    is_torch_cuda_available,
    is_torch_mps_available,
    get_str_time,
    )

# from .model_config import *
# from .runtime_config import *


def load_config(config_path=None)->[configparser.ConfigParser, dict]:
    import_tqdm()
    devices = get_available_devices()
    if "cuda" in devices:
        print("[+] using cuda to drive.")
    else:
        print("[+] using cpu to drive.")

    if config_path is None:
        return
    else:
        base_config = configparser.ConfigParser()
        base_config.read(config_path)
        config_dict = {section: dict(base_config.items(section)) for section in base_config.sections()}

        return base_config, config_dict

def load_model_card(model_card_path=""):
    if os.path.splitext(model_card_path)[1] == ".yaml":
        with open(model_card_path, "r") as f:
            model_card = yaml.safe_load(f)

    return model_card

def load_platform():
    return get_platform()

@lru_cache()
def load_root(*,depth=2)->str:
    # depth 是该文件相对于最顶层目录的深度，config文件相对于根目录有几个斜杠该数字就是几
    rtn = __file__
    while depth > 0:
        rtn = os.path.dirname(rtn)
        depth -= 1
    return rtn

def get_relative_path(abs_path, as_dir=False):
    if as_dir:
        return os.path.relpath(os.path.dirname(abs_path), load_root())
    else:
        return os.path.relpath(abs_path, load_root())

@lru_cache()
def get_available_devices() -> FrozenSet[str]:
    """
    Returns a frozenset of devices available for the current PyTorch installation.
    """
    devices = {"cpu"}  # `cpu` is always supported as a device in PyTorch

    if is_torch_cuda_available():
        devices.add("cuda")

    if is_torch_mps_available():
        devices.add("mps")

    # if is_torch_xpu_available():
    #     devices.add("xpu")
    #
    # if is_torch_npu_available():
    #     devices.add("npu")
    #
    # if is_torch_mlu_available():
    #     devices.add("mlu")
    #
    # if is_torch_musa_available():
    #     devices.add("musa")

    return frozenset(devices)