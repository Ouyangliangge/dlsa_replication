## Deep Learning-Based Statistical Arbitrage Strategy Replication

*replication of the paper "Deep Learning Statistical Arbitrage",available at https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3862004 and https://arxiv.org/abs/2106.04028.*

### Introduction

* Fama-French factor models for residuals computation
* Various machine learning models ( OUFFN, CNNTransformer, and FourierFFN) to construct and optimize trading strategies
* With the goal of maximizing risk-adjusted returns(Sharpe ratio and mean-var)
* Data from S&P 500, by yfinance to collect 10 stocks.

---

### Structure

The project is organized as follows:

* `famafrench.ipynb` contains the code for data collecting and creating residuals from raw data 

* `train_test.py` contains the code for training a trading policy model and simulating trading
* `train_test_2.py` contains the modified code from `train_test.py` (add excess returns to optimize the objective)
* `run_train_test.ipynb` is a user interface to `train_test.py` which deals with configuration, saving results, etc
* `preprocess.py` contains functions for preprocessing residual time series data into a form usable by a trading policy model
* `utils.py` contains helpful functions used throughout
* `results` should contains folders, which will contain the results of and plots for trading policy model tests conducted by `run_train_test.py`
* `models` contains code for trading policy models
* `config` contains configuration files which define various tests of trading policy models on residual time series
* `residuals` contains data generated from `famafrench.ipynb`

---

### Quick start

* First to run `famafrench.ipynb` to generate residuals, and will be saved in the `residuals` folder

* Then run `run_train_test.ipynb ` 

---

### FAQ

#### **1. Training is slow**:

* If you have access to a GPU, ensure PyTorch detects it properly, and set the `device` to `cuda` in the configuration files.

* You can also adjust the `batch_size` and `num_epochs` in the configuration files to speed up training.

#### **2. Failed to run `run_train_test.py`**:

* Make sure you have folders such as `/results/Unknown`, `/results/cumulative_returns`, `/results/turnovers` ,`/results/short_proportions`, `/results/Checkpoint`  at first
* Make sure `__init__.py` exists in the `models` folder，to ensure Python treats the `models` folder as a package

#### **3. RuntimeError**: 

* `Function 'StdBackward0' returned nan values in its 0th output.` 

  Make every `std=torch.std (__ + 1e-8 ) + 1e-8`

  

