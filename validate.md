# Validation

## Objective
Validate the test results for the three datasets mentioned in the project.

## MNIST
- The OOCS model, SM_CNN, and BaseNet0 models were run with the following commands:
    ```terminal
    python experiments/RobustnessMNIST/Robustness_MNIST_train.py --model OOCS --epochs 10 --lr 0.001
    python experiments/RobustnessMNIST/Robustness_MNIST_train.py --model SM --epochs 10 --lr 0.001
    python experiments/RobustnessMNIST/Robustness_MNIST_train.py --model Basenet --epochs 10 --lr 0.001
    ```
- The inversion formula for the MNIST dataset is simply:

```python
import numpy as np
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_test_dark = []
for x in x_test:
    x_test_dark.append(abs(x - 255.0))
x_test_dark = np.array(x_test_dark)
```

### MNIST Results -- OOCS
```text
epoch: 0, step:750, loss: 4.230811, accuracy: 0.538792
validation epoch: 0, loss: 4.240932, accuracy: 0.521937
epoch: 1, step:750, loss: 3.817282, accuracy: 0.882833
validation epoch: 1, loss: 3.829680, accuracy: 0.866896
epoch: 2, step:750, loss: 3.741464, accuracy: 0.933062
validation epoch: 2, loss: 3.748116, accuracy: 0.924792
epoch: 3, step:750, loss: 3.711000, accuracy: 0.950104
validation epoch: 3, loss: 3.716664, accuracy: 0.942167
epoch: 4, step:750, loss: 3.695693, accuracy: 0.959312
validation epoch: 4, loss: 3.700820, accuracy: 0.950437
epoch: 5, step:750, loss: 3.686946, accuracy: 0.962271
validation epoch: 5, loss: 3.690009, accuracy: 0.956625
epoch: 6, step:750, loss: 3.682875, accuracy: 0.964958
validation epoch: 6, loss: 3.686197, accuracy: 0.958750
epoch: 7, step:750, loss: 3.679478, accuracy: 0.967167
validation epoch: 7, loss: 3.682190, accuracy: 0.961396
epoch: 8, step:750, loss: 3.676400, accuracy: 0.969229
validation epoch: 8, loss: 3.679822, accuracy: 0.962188
epoch: 9, step:750, loss: 3.673940, accuracy: 0.970042
validation epoch: 9, loss: 3.676731, accuracy: 0.964125
test on original data, loss: 3.680455, accuracy: 0.961300
test on inverted data, loss: 3.819474, accuracy: 0.822200
```

### MNIST Results - SM-CNN
```text
epoch: 0, step:750, loss: 3.731590, accuracy: 0.930458
validation epoch: 0, loss: 3.737823, accuracy: 0.921938
epoch: 1, step:750, loss: 3.648454, accuracy: 0.986771
validation epoch: 1, loss: 3.653484, accuracy: 0.981083
epoch: 2, step:750, loss: 3.641194, accuracy: 0.990604
validation epoch: 2, loss: 3.647474, accuracy: 0.982979
epoch: 3, step:750, loss: 3.636082, accuracy: 0.994292
validation epoch: 3, loss: 3.643177, accuracy: 0.985542
epoch: 4, step:750, loss: 3.633148, accuracy: 0.995812
validation epoch: 4, loss: 3.641133, accuracy: 0.986354
epoch: 5, step:750, loss: 3.630601, accuracy: 0.996750
validation epoch: 5, loss: 3.638347, accuracy: 0.987979
epoch: 6, step:750, loss: 3.628738, accuracy: 0.997813
validation epoch: 6, loss: 3.637501, accuracy: 0.988417
epoch: 7, step:750, loss: 3.627805, accuracy: 0.998354
validation epoch: 7, loss: 3.636693, accuracy: 0.988583
epoch: 8, step:750, loss: 3.627049, accuracy: 0.998750
validation epoch: 8, loss: 3.636508, accuracy: 0.988500
epoch: 9, step:750, loss: 3.626228, accuracy: 0.999188
validation epoch: 9, loss: 3.635587, accuracy: 0.989125
test on original data, loss: 3.634464, accuracy: 0.989100
test on inverted data, loss: 4.333101, accuracy: 0.287600
```

### MNIST Results - Basenet
```Text
epoch: 0, step:750, loss: 4.178418, accuracy: 0.580854
validation epoch: 0, loss: 4.188151, accuracy: 0.570000
epoch: 1, step:750, loss: 3.975698, accuracy: 0.690729
validation epoch: 1, loss: 3.985812, accuracy: 0.682250
epoch: 2, step:750, loss: 3.914979, accuracy: 0.735812
validation epoch: 2, loss: 3.924569, accuracy: 0.725792
epoch: 3, step:750, loss: 3.767000, accuracy: 0.877458
validation epoch: 3, loss: 3.777560, accuracy: 0.865104
epoch: 4, step:750, loss: 3.750962, accuracy: 0.886792
validation epoch: 4, loss: 3.759204, accuracy: 0.876229
epoch: 5, step:750, loss: 3.673456, accuracy: 0.962812
validation epoch: 5, loss: 3.677286, accuracy: 0.957937
epoch: 6, step:750, loss: 3.651650, accuracy: 0.983333
validation epoch: 6, loss: 3.653483, accuracy: 0.980542
epoch: 7, step:750, loss: 3.649170, accuracy: 0.984771
validation epoch: 7, loss: 3.653033, accuracy: 0.979604
epoch: 8, step:750, loss: 3.647347, accuracy: 0.985854
validation epoch: 8, loss: 3.652103, accuracy: 0.979937
epoch: 9, step:750, loss: 3.645700, accuracy: 0.987063
validation epoch: 9, loss: 3.649907, accuracy: 0.981521
test on original data, loss: 3.646738, accuracy: 0.983200
test on inverted data, loss: 4.379572, accuracy: 0.243200
```
