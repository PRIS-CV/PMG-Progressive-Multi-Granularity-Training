
# Progressive Multi-Granularity Training
 
Code release for Fine-Grained Visual Classiﬁcation via Progressive Multi-Granularity Training of Jigsaw Patches (ECCV2020)
 
### Requirement
 
python 3.6

PyTorch >= 1.3.1

torchvision >= 0.4.2

### Training

1. Download datatsets for FGVC (e.g. CUB-200-2011, Standford Cars, FGVC-Aircraft, etc) and organize the structure as follows:
```
dataset
├── train
│   ├── class_001
|   |      ├── 1.jpg
|   |      ├── 2.jpg
|   |      └── ...
│   ├── class_002
|   |      ├── 1.jpg
|   |      ├── 2.jpg
|   |      └── ...
│   └── ...
└── test
    ├── class_001
    |      ├── 1.jpg
    |      ├── 2.jpg
    |      └── ...
    ├── class_002
    |      ├── 1.jpg
    |      ├── 2.jpg
    |      └── ...
    └── ...
```

2. Train from scratch with ``train.py``.


### Citation
 
Please cite our paper if you use PMG in your work.
```
@InProceedings{du2020fine,
  title={Fine-Grained Visual Classification via Progressive Multi-Granularity Training of Jigsaw Patches},
  author={Du, Ruoyi and Chang, Dongliang and Bhunia, Ayan Kumar and Xie, Jiyang and Song, Yi-Zhe and Ma, Zhanyu and Guo, Jun},
  booktitle = {European Conference on Computer Vision},
  year={2020}
}

```

## Contact
Thanks for your attention!
If you have any suggestion or question, you can leave a message here or contact us directly:
- mazhanyu@bupt.edu.cn
- beiyoudry@bupt.edu.cn

