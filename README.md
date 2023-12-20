# PriSA
Pytorch implementation for Multimodal Sentiment Analysis with Preferential Fusion and Distance-aware Contrastive Learning (ICME 2023 Oral).


# Dependencies
## Environments
We provide the anaconda enviroments to help you build a runnable environment.
```bash
conda env create -f environments.yml
conda activate prisa
```
## Datasets
Please download [MOSEI dataset](https://share.weiyun.com/6zj86Uhn) into `./data` for training and evaluation.

## Pretrained Models
Please download [bert-base-uncased](https://huggingface.co/bert-base-uncased) from huggingface into `./pretrained_models`.

# Training
You can train the model using the following command. The output will be saved at `/tmp/log/`, you can modify `.runx` to change the path of training log.
```bash
python -m runx.runx mosei.yml -i
```

Feel free to concat with us (mafp@foxmail.com) if you have any problem.

# Citation
```
@inproceedings{ma2023multimodal,
  title={Multimodal Sentiment Analysis with Preferential Fusion and Distance-aware Contrastive Learning},
  author={Ma, Feipeng and Zhang, Yueyi and Sun, Xiaoyan},
  booktitle={2023 IEEE International Conference on Multimedia and Expo (ICME)},
  pages={1367--1372},
  year={2023},
  organization={IEEE}
}
```
