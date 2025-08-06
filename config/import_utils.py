from packaging import version
import importlib.machinery
import importlib.metadata
import importlib.util
import sys
import os
import datetime
import platform
from torch import cuda
from typing import Any, Dict, FrozenSet, Optional, Set, Tuple, Union
from functools import lru_cache

# come from transformer
# whether if it is useful, save here
def _is_package_available(pkg_name: str, return_version: bool = False) -> Union[Tuple[bool, str], bool]:
    # Check if the package spec exists and grab its version to avoid importing a local directory
    package_exists = importlib.util.find_spec(pkg_name) is not None
    package_version = "N/A"
    if package_exists:
        try:
            # Primary method to get the package version
            package_version = importlib.metadata.version(pkg_name)
        except importlib.metadata.PackageNotFoundError:
            # Fallback method: Only for "torch" and versions containing "dev"
            if pkg_name == "torch":
                try:
                    package = importlib.import_module(pkg_name)
                    temp_version = getattr(package, "__version__", "N/A")
                    # Check if the version contains "dev"
                    if "dev" in temp_version:
                        package_version = temp_version
                        package_exists = True
                    else:
                        package_exists = False
                except ImportError:
                    # If the package can't be imported, it's not available
                    package_exists = False
            else:
                # For packages other than "torch", don't attempt the fallback and set as not available
                package_exists = False
        # logger.debug(f"Detected {pkg_name} version: {package_version}")
    if return_version:
        return package_exists, package_version
    else:
        return package_exists

_torch_available, _torch_version = _is_package_available("torch", return_version=True)

def get_platform():
    system_platform = platform.system()
    return system_platform

def is_jupyter():
    if 'ipykernel' in sys.modules:
        return True
    else:
        return False

def import_tqdm():
    #  判断是否是notebook环境. 以选择正确的tqdm
    #  -- tqdm自己提供了auto方法
    # if is_jupyter():
    #     from tqdm.notebook import tqdm
    # else:
    #     from tqdm import tqdm
    from tqdm.auto import tqdm
    pass

def is_torch_available():
    return _torch_available

def is_torch_cuda_available():
    if is_torch_available():
        import torch

        return torch.cuda.is_available()
    else:
        return False

def is_torch_mps_available(min_version: Optional[str] = None):
    if is_torch_available():
        import torch

        if hasattr(torch.backends, "mps"):
            backend_available = torch.backends.mps.is_available() and torch.backends.mps.is_built()
            if min_version is not None:
                flag = version.parse(_torch_version) >= version.parse(min_version)
                backend_available = backend_available and flag
            return backend_available
    return False

def get_str_time(Date=True, Time=True, dateDiv="-", timeDiv="-", datetimeDiv="_"):
    format_str = ""
    if Date:
        format_str += f"%Y{dateDiv}%m{dateDiv}%d"
    if Date and Time:
        format_str += datetimeDiv
    if Time:
        format_str += f"%H{timeDiv}%M{timeDiv}%S"
    return datetime.datetime.now().strftime(format_str)








# if system_platform == "Darwin":
#     database_path = None
# elif system_platform == "Linux":
#     database_path = os.path.join(os.path.expanduser(os.path.join("~", "Data")), "FW.sqlite")
# elif system_platform == "Windows":
#     database_path = os.path.join(os.path.expanduser(os.path.join("~", "Data")), "FW.sqlite")






