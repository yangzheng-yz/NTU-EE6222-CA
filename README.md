# A Method Using Image Edge Information for Action Recognition in the Dark

This folder includes the whole code and scripts mentioned in the CA report.

## Prerequisites

This code is tested on PyTorch 1.8.0+cu111.
The whole environment requirements are provided in requirements.yaml. The following command is provided to copy the conda environment into your own computer/server.
```
conda env create -f requirements.yaml
```

## Training

Train with initialization from pre-trained models:
```
python train_arid11.py --network resnet --batch-size 8 --end-epoch 50 --save-frequency 5 --brighten_method lime --task-name lime_train
```
or you can run 
```
sh run.sh
```

the pretrained model is provided in */ARID_v1/network/pretrained/r3d_18-b3b3357e.pth*

## Testing

### Evaluate the trained model without fusion:
```
cd test
sh test.sh
```
you should use the argument *-mode train/test/val* to determine on which dataset you want to evaluate.
*--brighten-method* is used to determine on which argumented dataset you want to evaluate. Here we mainly test *--brighten-method lime* and *--brighten-method lime_canny* which are two streams mentioned in the paper.
These two methods' corresponding weights are provided in **"exps/exps_r3d18/models/archive_lime-arid-resnet3d-18-orimgsample-is-not-dark-std-mean/best.pth"** and **"exps/models/archive_arid-resnet3d-18-MGsample-canny-is-not-dark-weights_1.3_0.5-canny_thresh_100_200/best.pth"**.

### Evaluate the trained model with fusion:
```
cd test
sh test_fusion.sh
```
you should use the argument *-mode train/test/val* to determine on which dataset you want to evaluate.
Note that *--beighten-method2* and *--model2-path* refer to edge dataset and the weights trained on edge dataset repectively.


## Other Information



- This code base is adapted from [Multi-Fiber Network for Video Recognition](https://github.com/xuyu0010/ARID_v1), I would like to thank the authors for providing the code base.
- You may contact me through e220193@e.ntu.edu.sg
