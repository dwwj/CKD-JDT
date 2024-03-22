#数据预处理
cd tools && bash get_mot_17.sh && cd ..

#训练
cd src/
CUDA_VISIBLE_DEVICES=0 python main.py tracking --exp_id mot17_half --dataset mot --dataset_version 17halftrain --num_classes 1 --pre_hm --ltrb_amodal --same_aug --hm_disturb 0.05 --lost_disturb 0.4 --fp_disturb 0.1 --gpus 0  --batch_size 16 --kd_weight 0.2 --input_w 640 --input_h 640
#测试
CUDA_VISIBLE_DEVICES=0 python test.py tracking --exp_id mot17_half --dataset mot --dataset_version 17halfval --pre_hm --ltrb_amodal --track_thresh 0.4 --pre_thresh 0.5 --load_model ../exp/tracking/mot17_half/model_last.pth --batch_size 1 --input_w 1088 --input_h 608
cd ..