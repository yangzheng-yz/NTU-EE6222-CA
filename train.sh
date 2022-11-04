CUDA_VISIBLE_DEVICES=0 python train_arid11.py --network resnet --batch-size 8 \
--end-epoch 50 --save-frequency 5 --task-name lime_arid-r3d18_for_checkbatch8-mgsampler-is_not_dark \
--brighten_method lime
# --is-dark