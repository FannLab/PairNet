# PNL

PairNet algorithm used to build polygenic risk score in PNL method.

## Installation

```bash
$ pip install git+https://github.com/FannLab/PairNet.git
```

## Usage

In the repo, there are a few points to note:


First, make sure that your device has CUDA properly installed, as this package utilizes CUDA.

You need to import the PairNet package using the following statement:

```
import PairNet
```

In the PairNet package, there are two available functions: `PairNet.PairNet.train` and `PairNet.PairNetClassifier.PairNetClassifier`. The `PairNetClassifier` is the model of PairNet, intended for users familiar with deep learning.

The `train` function is used to execute the complete training process and ultimately generate the ROC (Receiver Operating Characteristic) curves.

Here is an introduction to the parameters of the `train` function:

```
PairNet.PairNet.train(train_z, train_p, val_z, val_p, test_z, test_p,OUT_DIR,SEED=None,EPOCHS = 20,MODEL_DIR = None,log_file = None)
```

`train_z, val_z, test_z`: Type: numpy.ndarray; Z matrix. It should be split into three sets: train, test, and validation.

`train_p, val_p, test_p`: Type: numpy.ndarray; Corresponding to the Z matrix, these are the labels for the traits. Please use 0, 1, or 1, 2 to indicate the traits.

`OUT_DIR`: Location for storing the ROC (Receiver Operating Characteristic) curves.

`SEED`: Input an integer to fix the random seed, or leave it as None to not fix the random seed.

`EPOCHS`: Number of iterations for training.

`MODEL_DIR`: Location for storing the trained model. If set to None, the model will not be saved.

`log_file`: Location for storing the simple data of the Z matrix. If set to None, the data will not be saved.

Please make sure to set these variables according to your requirements.

## Test Data

```
import pandas as pd
import numpy as np

sample_file = 'data/split/demo_1_sample.txt'
genotype_file = 'data/split/demo_1_Z.txt'
train_sample_file = 'data/split/demo_1_train_sample.txt'
test_sample_file = 'data/split/demo_1_test_sample.txt'
val_sample_file = 'data/split/demo_1_val_sample.txt'

sample_all = pd.read_csv(sample_file)
samples = pd.read_csv(train_sample_file,sep=' ').FID.tolist()
train_sample_idx = np.flatnonzero(sample_all.FID.isin(samples))

samples = pd.read_csv(test_sample_file,sep=' ').FID.tolist()
test_sample_idx = np.flatnonzero(sample_all.FID.isin(samples))

samples = pd.read_csv(val_sample_file,sep=' ').FID.tolist()
val_sample_idx = np.flatnonzero(sample_all.FID.isin(samples))

skipcols = ["FID","PHENO"]
data = pd.read_csv(genotype_file, usecols=lambda x: x not in skipcols).to_numpy()

col_idx = range(0, data.shape[1])
x_train = data[np.ix_(train_sample_idx, col_idx)]
y_train = sample_all.PHENO.iloc[train_sample_idx].values - 1

x_val = data[np.ix_(val_sample_idx, col_idx)]
y_val = sample_all.PHENO.iloc[val_sample_idx].values - 1

x_test = data[np.ix_(test_sample_idx, col_idx)]
y_test = sample_all.PHENO.iloc[test_sample_idx].values - 1

import PairNet

PairNet.train_val_test.train(x_train,y_train,x_val,y_val,x_test,y_test,'out/test')
```

## License

`PairNet` was created by syuepu. It is licensed under the terms of the MIT license.

## Credits

`PairNet` was created with [`cookiecutter`](https://cookiecutter.readthedocs.io/en/latest/) and the `py-pkgs-cookiecutter` [template](https://github.com/py-pkgs/py-pkgs-cookiecutter).


### Citation

Jhang, Y. J., Chu, Y. C., Tai, T. M., Hwang, W. J., Cheng, P. W., & Lee, C. K. (2019, July). Sensor based dynamic hand gesture recognition by PairNet. In _2019 International Conference on Internet of Things (iThings) and IEEE Green Computing and Communications (GreenCom) and IEEE Cyber, Physical and Social Computing (CPSCom) and IEEE Smart Data (SmartData)_ (pp. 994-1001). IEEE.
https://ieeexplore.ieee.org/document/8875280

