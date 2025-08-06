
## Repository Agenda

1. [Introduction](#Introduction)

2. [Installation & Setup](#Installation)

3. [Training and Evaluation](#Train)

4. [Repository Structure](#FileStructure)

5. [Data Availability](#Data)

6. [Acknowledgements](#Acknowledgements)

## Introduction

Forecasting target trajectory without spatial constraints holds enormous value in a wide spread of scientific and engineering applications. However, this task poses significant challenges due to the tremendous protential choices of target movement in large free space. 
From physical modeling grounded in fundamental laws to machine learning centered on semantic relevance, academia and industry have presented diverse methods to address this complications.
We introduce ConvFIS, a priori-free trainable rule-based fuzzy inference learning framework that builds chronological matches for long-term trajectories in open space. Futhermore, these matched patterns, termed rules, can be extracted and analysed to provide model interpretability as well as to inform the laws of trajectory forecasting. The proposed rule analysis methodology provides a new view on chaotic and complex giant parameter models. In particular, temporal symmetry breaking is observed in the rule consequences before and after training, which reveals a physical characterization of how training influences parameters.
Benchmarked against the most advanced spatio-temporal predictors, ConvFIS maintained superior long-term forecasting accuracy while favorable reduced computational demands.

## Installation

To set up the repository, follow the steps below:

* Clone the repository: `git clone https://github.com/Kinddle_tick/ConvFIS.git`.

## Train
Run main_train_[model_name].py to train the model and the results will be saved in the output folder

> [!TIP] 
> You can adjust the number of training epoches, seed, etc. by changing the configuration in default.ini.



## FileStructure

We use python software designed for parameter management and training

The detailed structure of this project is shown as follows:

```
ConvFIS-main/
├── config                                          -- config files folder
├── data_source                                     -- python package for data read
├── frame/
│   ├── data_process/                               -- tools to process data
│   ├── eval_process/                               -- tools to analyze result
│   │   ├── painter                                 -- functions to draw picture                                
│   │   ├── error_fuction.py                        -- Several functions for calculating the error 
│   │   └── eval_analyze.py                         -- use class EvalHandler to analyze data
│   ├── painter_format.py                           -- config for picture drawing
│   ├── reporter.py                                 -- Director of Human-Computer Interaction
│   ├── trainer.py                                  -- Classes used to guide training            
│   ├── trainer_callback.py                         -- Callback Functional Interaction Methods
│   └── training_args.py                            -- Parameter management for training
├── model/
│   ├── fuzzy_inference                             -- Our model       
│   ├── gpt_model                                   -- GPT2
│   ├── resnet_model                                -- Resnet
│   ├── timeseries_model                            -- timeseries model from https://github.com/thuml/Time-Series-Library
├── analyze_track_feature.py                        -- Exploring diffusion properties
├── default.ini                                     -- Management of various model training parameters
├── main_data_flight.py                             -- Data processing scripts
├── main_eval_common.py                             -- Test Scripts - General
├── main_eval_compare_all.py                        -- Test Scripts - For all different model
├── main_eval_fis.py                                -- Test Scripts - For one or more ConvFis
├── main_eval_fis_mult_epoch.py                     -- Test Scripts - For all ckpt of one model
├── main_eval_fis_rule_num.py                       -- Test Scripts - For ConvFis in different number of rules
├── main_train_ConvFIS.py                           -- Train Scripts - ConvFIS
├── main_train_Dlinear.py                           -- Train Scripts - Dlinear
├── main_train_GPT2.py                              -- Train Scripts - GPT-2
├── main_train_LSTM.py                              -- Train Scripts - LSTM
├── main_train_Transformer.py                       -- Train Scripts - Transformer
├── main_train_iTransformer.py                      -- Train Scripts - iTransformer
├── readme.md                                       -- this file
├── script_mk_gif.py                                -- Work with “main_eval_fis_mult_epoch.py” to generate a dynamic graph
└── support_config_parser.py                        -- Implement parameter management in outer fit 'default.ini'
```


## Data

The original training and evaluation data that we presented in the paper are avaiable at [here](source/data_group/).


## Acknowledgements

* Project file in [model/timeseries_model](model/timeseries_model) are based on same files from [Time-Series-Library](https://github.com/thuml/Time-Series-Library).
