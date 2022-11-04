# CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=1 
python evaluate_video.py --gpus 0 --save-result --mode val \
--task-name lime-arid-resnet3d-18-orimgsample-is-not-dark-std-mean \
--network resnet --batch-size 1 \
--brighten-method lime \
--model-path /media/mldadmin/home/s122mdg36_01/arid/assignment1/solutions/ARID_v1/exps/exps_r3d18/models/archive_lime-arid-resnet3d-18-orimgsample-is-not-dark-std-mean/best.pth
# --model2-path /media/mldadmin/home/s122mdg36_01/arid/assignment1/solutions/ARID_v1/exps/models/archive_arid-resnet3d-18-MGsample-canny-is-not-dark-weights_1.3_0.5-canny_thresh_100_200/best.pth 
# --model-path /media/mldadmin/home/s122mdg36_01/arid/assignment1/solutions/ARID_v1/exps/models/archive_arid-resnet3d-18-MGsample-canny-is-not-dark-weights_1.3_0.5-canny_thresh_100_200/best.pth
# --is-dark
# --brighten-method lime_canny_100_200 \