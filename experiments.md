# Experiments

## 1. Effect of neuron number from Dense Layer:
Run the following number of neurons:
```text
[
    10 -- done
    32 -- done
    64 -- done
    100 -- done
    128 -- done
    256 -- done
    512
]
```
### Results
```Python
python experiments/RobustnessMNIST/Robustness_MNIST_train.py --model OOCS --epochs 5 --lr 0.01 --save_name control --num_neurons 100

python experiments/RobustnessMNIST/Robustness_MNIST_train.py --model OOCS --epochs 5 --lr 0.01 --save_name n1 --num_neurons 1
```