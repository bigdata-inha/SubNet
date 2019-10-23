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
      <th rowspan=2>subset</th>
      <th colspan=2>subNetwork</th>
      <th colspan=2>OriginalNetwork</th>
      <th rowspan=2>A/B (%)</th>
     </tr>
     <tr> 
      <th># Params (A)</th>
      <th>Test - acc</th>
      <th># Params (B)</th>
      <th>Test - acc</th>
     </tr>
    </thead> 
    <tbody align='center'> 
     <tr> 
      <td colspan=6>Using MNISTdataset Network</td> 
     </tr>
     <tr> 
      <td>[1, 4]</td>
      <td>21,251</td>
      <td>0.999</td>
      <td rowspan=4>124,825</td>
      <td>0.983</td>
      <td>17.02</td>
     </tr>
     <tr> 
      <td>[0, 2, 6]</td>
      <td>29,947</td>
      <td>0.992</td>
      <td>0.980</td>
      <td>23.99</td>
     </tr>
     <tr> 
      <td>[0, 4, 6, 7]</td>
      <td>40,243</td>
      <td>0.992</td>
      <td>0.981</td>
      <td>32.24</td>
     </tr>
     <tr> 
      <td>ALL</td>
      <td>124,825</td>
      <td>0.979</td>
      <td>0.979</td>
      <td>100.</td>
     </tr>
     <tr> 
      <td colspan=6>Using FashionMNISTdataset Network</td> 
     </tr>
     <tr> 
      <td>subset for shoes categories [5, 7, 9]</td>
      <td>87,147</td>
      <td>0.963</td>
      <td rowspan=2>330,670</td>
      <td>0.962</td>
      <td>26.35</td>
     </tr>
     <tr> 
      <td>ALL</td>
      <td>330,670</td>
      <td>0.911</td>
      <td>0.911</td>
      <td>100.</td>
     </tr>
    </tbody> 
</table>

### Average number of nodes according to number of subset elements
The following results can be reproduced with command:

    python main.py --dataset MNIST --subset 0 1 2 3 4 5 6 7 8 9 --meanNodes True
    python main.py --dataset FashionMNIST --subset 0 1 2 3 4 5 6 7 8 9 --meanNodes True
    
<table align='center'>
<tr align='center'>
<td> Average number of Nodes </td>
</tr>
<tr>
<td><img src = 'images/Average number of nodes.png' height = '400px'></td>
</tr>
<tr align='center'>
<td>X-axis: Average number of nodes, Y-aixs: number of subset elements</td>
</tr>
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
* `--subset`:  Subset elements of total categories. *example*: `--subset 0 1 2`

*Optional*: 
* `--meanNodes`: Whether or not to print the average number of nodes. *type*: `bool`, *Default*: `False`

## Acknowledgements
This implementation has been tested with Keras 2.2.4 on Windows 10.
