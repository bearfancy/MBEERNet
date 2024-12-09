# MBEERNet
## Requirements
- colorama==0.4.6
- einops==0.8.0
- matplotlib==3.7.2
- numpy==1.25.0
- rich==13.9.4
- scikit_learn==1.3.1
- scipy==1.14.1
- thop==0.1.1.post2209072238
- torch==2.0.0
- torchinfo==1.8.0
- tqdm==4.65.0

## 创建运行环境
'''bash:
  conda create -n MBEER python==3.10.12
  pip install -r requirements.txt
'''






## Installation
Get the code.
```bash
git clone https://github.com/rhett-chen/graspness_implementation.git
cd graspnet-graspness
```
Install packages via Pip.
```bash
pip install -r requirements.txt
```
Compile and install pointnet2 operators (code adapted from [votenet](https://github.com/facebookresearch/votenet)).
```bash
cd pointnet2
python setup.py install
```
Compile and install knn operator (code adapted from [pytorch_knn_cuda](https://github.com/chrischoy/pytorch_knn_cuda)).
```bash
cd knn
python setup.py install
```
Install graspnetAPI for evaluation.
```bash
git clone https://github.com/graspnet/graspnetAPI.git
cd graspnetAPI
pip install .
```
