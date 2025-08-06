#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :__init__.py.py
# @Time      :2023/6/14 6:15 PM
# @Author    :Oliver
# import FuzzyInferenceSystem.Layers.Fuzzifier as Fuzzifier
# import FuzzyInferenceSystem.Layers.Defuzzifier as Defuzzifier
# import FuzzyInferenceSystem.Layers.Inference as Inference

from . import Fuzzifier
from . import Defuzzifier
from . import Inference

# from .Amplifier import SingletonAmplifier
# from .Defuzzifier import SingletonHeightDefuzzifierLayer,SingletonTskDefuzzifierLayer,SingletonPairTskDefuzzifierLayer
from .Defuzzifier import SingletonHeightDefuzzifierLayer,SingletonTskDefuzzifierLayer
from .Fuzzifier import SingletonFuzzifierLayer, TimeFuzzifierLayer
from .Inference import (SlopeMappingInferenceLayer,
                        GaussianMappingInferenceLayer,
                        TrapezoidalMappingInferenceLayer,
                        StrictlyTrapezoidalMappingInferenceLayer)
from .Inference import (MultGaussianInferenceLayer,
                        MultChannelGaussianInferenceLayer,
                        MultiTimeGaussianInferenceLayer,
                        TimeGaussianInferenceLayer)
# from .GenDefuzzifier import SingletonGenDefuzzifierLayer

# import FuzzyInferenceSystem.Layers.Loss as Loss

# from FuzzyInferenceSystem.Layers.Amplifier import SingletonAmplifier
#
# from FuzzyInferenceSystem.Layers.Defuzzifier import SingletonHeightDefuzzifierLayer, SingletonTskDefuzzifierLayer
#
# from FuzzyInferenceSystem.Layers.Fuzzifier import SingletonFuzzifierLayer
#
# from FuzzyInferenceSystem.Layers.Inference import TrapInferenceLayer,GaussianInferenceLayer,HalfTrapInferenceLayer,\
#     StrictlyTrapInferenceLayer,MultGaussianInferenceLayer,MultChannelGaussianInferenceLayer

# from FuzzyInferenceSystem.Layers.Loss import MSELoss, RMSELoss,EuclideanLoss

# from FuzzyInferenceSystem.Layers.Defuzzifier import TskDefuzzifierLayer,HeightDefuzzifierLayer
# from FuzzyInferenceSystem.Layers.Fuzzifier import FuzzifierLayer
# from FuzzyInferenceSystem.Layers.Amplifier import Amplifier
"""
生成一些基本的层
"""

