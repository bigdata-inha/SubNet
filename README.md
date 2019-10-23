# SubNet
Keras implementation of SubNetwork

## Results
### Compare accuracy and number of parameters
The following results can be reproduced with command:
    
    python main.py --dataset MNIST --subset 1 4
    python main.py --dataset MNIST --subset 0 2 6
    python main.py --dataset MNIST --subset 0 4 6 7
    python main.py --dataset MNIST --subset 0 1 2 3 4 5 6 7 8 9
    python main.py --dataset FashionMNIST --subset 5 7 9
    python main.py --dataset FashionMNIST --subset 0 1 2 3 4 5 6 7 8 9

<table> 
    <thead> 
     <tr> 
      <th>Layer 1</th> 
      <th>Layer 2</th> 
      <th>Layer 3</th> 
     </tr> 
    </thead> 
    <tbody> 
     <tr> 
      <td rowspan=4>L1 Name</td> 
      <td rowspan=2>L2 Name A</td> 
      <td>L3 Name A</td> 
     </tr> 
     <tr> 
      <td>L3 Name B</td> 
     </tr> 
     <tr> 
      <td rowspan=2>L2 Name B</td> 
      <td>L3 Name C</td> 
     </tr> 
     <tr> 
      <td>L3 Name D</td> 
     </tr> 
    </tbody> 
</table> 

## Usage
### Prerequisites
1. Keras
2. Python packages: numpy
### Command
    python main.py --dataset <choose dataset> --subset <subset of total categories>
*Example*: `python main.py --dataset MNIST --subset 0 1 2 3`

### Arguments
*Required*:
* `--dataset`: Choose datset. *Option*: `MNIST` or `FasionMNIST`
* `--subset`: Elements of subset of total categories. *example*: `--subset 0 1 2`

*Optional*: 
* `--meanNodes`: Whether or not to print the average number of nodes. *Default*: `False`

## Acknowledgements
This implementation has been tested with Keras 2.2.4 on Windows 10.
