# SubNet
Keras implementation of SubNetwork

## Results

## Usage
### Prerequisites
1. Keras
2. Python packages: numpy
### Command
    python main.py --dataset <choose dataset> --subset <subset of total categories>
*Example*: python main.py --dataset MNIST --subset 0 1 2 3

### Arguments
*Required*:
* `--dataset`: Choose datset. *Option*:`MNIST`or `FasionMNIST`
* `--subset`: Elements of subset of total categories. *example*: `--subset 0 1 2`

*Optional*: 
* `--meanNodes`: Whether mean of number of nodes. *Default*: `False`

## Acknowledgements
This implementation has been tested with Keras 2.2.4 on Windows 10.
