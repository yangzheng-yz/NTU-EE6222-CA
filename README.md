# A Method Using Image Edge Information for Action Recognition in the Dark

This folder includes the whole code and scripts mentioned in the CA report. If this version README cannot help you implement my code, please refer to https://github.com/yangzheng-yz/NTU-EE6222-CA.git for up-to-date README.md.

## Prerequisites

This code is tested on PyTorch 1.8.0+cu111.
The whole environment requirements are provided in requirements.yaml. The following command is provided to copy the conda environment into your own computer/server.
```
conda env create -f requirements.yaml
```

## Folder Tree
refer to ./folder_tree.png.
Need to notice that under train or train_lime folder you should have several folder named 'Drink','Jump'... to save .mp4 files. Under val or test or val_lime or test_lime folder, you will have .mp4 files.

## data preprocessing
You can use dataset/ARID/scripts/brighten.py to generate enhanced video. It is required to modify the method you need in the code.

Also, you can use dataset/ARID/scripts/extract_edge.py to generate edge videos.

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
```
cd test
```

**One thing you need to notice firstly: if you use fusion methods (use two models) to evaluate, you need to make sure that your --brighten-method2 and --model2-path refer to the canny stream's enhancement method and models**

**Also remember if you need to use --brighten-method OR --brighten-methods AND --mode to refer to which dataset should be load and evaluated. For example, if you want to evaluate canny model on validation dataset without fusion, in your command line, you should have --mode val --brighten-method lime_canny. Then the code can search val_lime_canny dataset folder under ../dataset/ARID/**

### Evaluate the trained model without fusion:
```
CUDA_VISIBLE_DEVICES=0 python evaluate_video.py --brighten-mothod $lime OR ying_caip OR canny_50_150$ \
--model-path $PATH/TO/xxx.pth --mode $val OR test$
```
You should use the argument *-mode train/test/val* to determine on which dataset you want to evaluate.
*--brighten-method* is used to determine on which argumented dataset you want to evaluate. Here we mainly test *--brighten-method ying_caip* and *--brighten-method lime_canny_50_150* which are two streams mentioned in the paper.
These two methods' corresponding weights are provided in **"exps/tidy-logs-models/models/best-model_yingcaip_resnet3d_MGSampler.pth"** and **"exps/tidy-logs-models/models/best-model_lime-canny-thresh-50-150_resnet3d_MGsampler.pth"**. you should use the argument *-mode train/test/val* to determine on which dataset you want to evaluate.

### Evaluate the trained model with fusion:
```
CUDA_VISIBLE_DEVICES=0 python evaluate_video.py --brighten-mothod ying_caip \
--model-path $PATH/TO/best-model_yingcaip_resnet3d_MGSampler.pth --mode $val OR test$ \
--brighten-method2 lime_canny_50_150 --model2-path \
$PATH/TO/best-model_lime-canny-thresh-50-150_resnet3d_MGsampler.pth \
--fusion-mode $0 OR 1 OR 2$
```
Note that *--brighten-method2* and *--model2-path* MUST refer to edge dataset and the weights trained on edge dataset repectively. In addition, make sure that there are MLP512.pth AND MLP1024.pth in the same folder as evaluate_video.py. The inference result '$mode_output.txt' will be save the same folder as evaluate_video.py. The corresponding log file will be saved to ../exps/logs_test/xxxxxx.txt.


## Other Information



- This code base is adapted from [Multi-Fiber Network for Video Recognition](https://github.com/xuyu0010/ARID_v1), I would like to thank the authors for providing the code base.
- You may contact me through e220193@e.ntu.edu.sg
