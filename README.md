# New Paradigm of Adversarial Training: Breaking Inherent Trade-Off between Accuracy and Robustness via Dummy Classes


## 1. Dependencies

```
conda env create -f DUCAT.yaml
```


## 2. Training options

- `<DATASET>`: {`cifar10`,`cifar100`,`tinyimagenet`}
- `<ADV_TRAIN OPTION>`: {`adv_train`,`adv_trades`,`adv_mart`}
- `<CONSISTENCY_AT>`: {`consistency`}


## 3. Training code

```
# Example for PGD-AT + DUCAT 
python train.py --mode adv_train --dummy --epochs 130 --dataset cifar10 --augment_type base

# Example for MART + DUCAT 
python train.py --mode adv_mart --dummy --epochs 130 --dataset cifar10 --augment_type base

# Example for Consistency-AT + DUCAT 
python train.py --mode adv_train --consistency --dummy --epochs 130 --dataset cifar10 --augment_type base
```
