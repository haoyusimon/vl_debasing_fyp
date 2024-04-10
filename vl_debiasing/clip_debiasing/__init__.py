import pathlib
from typing import Any

class Dotdict(dict):
    def __getattr__(self, __name: str) -> Any:
        return super().get(__name)

    def __setattr__(self, __name: str, __value: Any) -> None:
        return super().__setitem__(__name, __value)

    def __delattr__(self, __name: str) -> None:
        return super().__delitem__(__name)

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, d):
        self.__dict__.update(d)


# DATA_PATH = (pathlib.Path(__file__) / ".." / ".." / "data").resolve().absolute()
# FAIRFACE_DATA_PATH = "/home/h/haoyu/CLIP_debiasing/fairface_margin125"
# PROMPT_DATA_PATH = DATA_PATH / "prompt_templates.csv"

DATA_PATH = "/home/h/haoyu/data"
FAIRFACE_DATA_PATH = "/home/h/haoyu/data/fairface"
PROMPT_DATA_PATH = "/home/h/haoyu/data/prompt_templates.csv"
FACET_DATA_PATH = "/home/h/haoyu/CLIP_debiasing/FACET"

from .models import *
from .measure_bias import measure_bias
