# Multi-Task-Learning

This is an implementation of exploiting the generalized mean for per-task loss aggregation in multi-task learning.
Our code is mainly based on [LibMTL](https://github.com/median-research-group/LibMTL?tab=readme-ov-file).
 

## Getting started

1. Create a virtual environment
   
   ```shell
   conda create -n gemtl python=3.8
   conda activate gemtl
   pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
   ```

2. Clone this repository

3. Install `LibMTL`
   
   ```shell
   cd GeMTL
   pip install -e .
   ```

## Requirements

- Python >= 3.8
- Pytorch >= 1.8.1

```shell
pip install -r requirements.txt
```
Please ensure all dependencies are installed to avoid compatibility issues during execution.

## Run


```shell
cd ./examples/{nyusp, office}
bash run.sh
```
You can modify run.sh scripts to experiment with different datasets or training configurations.
Detailed logging is provided during training to help monitor performance across tasks.

---
Pretrained models and example outputs will be shared in future updates.

## License

This repository is released under the [GPL-3.0](./LICENSE) license.
