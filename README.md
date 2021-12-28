# ECCV2020: Rethinking Image Deraining via Rain Streaks and Vapors

This is an implementation of the ASV-Joint model proposed in the ECCV 2020 paper
([Rethinking Image Deraining via Rain Streaks and Vapors](https://arxiv.org/pdf/2008.00823.pdf))
with PyTorch

# Requirements
python:3.6.4
pytorch: 0.4.1
ipdb
torchnet

## Evaluation

download the checkpoints file from [icloud](https://www.icloud.com.cn/iclouddrive/074yGL4RH3bydyd_-ni2vSZKw#eccv2020%5Fderain%5Fcheckpoints) and unzip them in the current directory (replace partial data files from github)

Run the following commands:
bash test.sh

you can set testing root in 'tesh.sh'


# Citation
    
    @inproceedings{wang-eccv20-rethinking,
      title={Rethinking Image Deraining via Rain Streaks and Vapors},
      author={Wang, Yinglong and Song, Yibing and Ma, Chao and Zeng, Bing},
      booktitle=ECCV,
      year={2020}
    }
