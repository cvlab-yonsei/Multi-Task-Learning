# GeMTL

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


## Dataset

You can download datasets in the following links.
- [NYUv2](https://github.com/lorenmt/mtan)  
- [Office-Home](https://www.hemanthdv.org/officeHomeDataset.html)


## Run

Training and testing codes are in `./examples/{nyusp, office}/main.py`.  
You can check the results by running the following command.

```shell
cd ./examples/{nyusp, office}
bash run.sh
```

## Reference

Our implementation is developed on the following repositories. Thanks to the contributors!
- [LibMTL](https://github.com/median-research-group/LibMTL?tab=readme-ov-file)  
- [CAGrad](https://github.com/Cranial-XIX/CAGrad)  
- [mtan](https://github.com/lorenmt/mtan)


## License

This repository is released under the [MIT](./LICENSE) license.
