This repository contains the official implementation for the manuscript: 

> [Decentralized Directed Collaboration for Personalized Federated Learning](https://arxiv.org/pdf/2405.17876)

# Experiments

The implementations of each method are provided in the folder `/fedml_api/standalone`, while experiments are provided in the folder `/fedml_experiments/standalone`.


Use the dataset corresponding bash file to run the experiments.

```
cd /fedml_experiments/standalone/dfedpgp
sh cifar10.sh
```

This code is based on the project in [DisPFL](https://github.com/rong-dai/DisPFL). 


# Citation

If you find this repo useful for your research, please consider citing the paper

```
@InProceedings{liu2024decentralized,
  title = 	 {Decentralized Directed Collaboration for Personalized Federated Learning},
  author =       {Liu, Yingqi and Shi, Yifan and Li, Qinglun and Wu, Baoyuan and Wang, Xueqian and Shen, Li},
  booktitle = 	 {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages = 	 {23168--23178},
  year = 	 {2024},
}

```
