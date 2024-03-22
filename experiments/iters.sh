list1=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0)


for i in ${list1[*]}
do
    
    echo $i
    #训练代码
    CUDA_VISIBLE_DEVICES=0 python main.py tracking --exp_id mot17_half-$i --dataset mot --dataset_version 17halftrain --num_classes 1 --pre_hm --ltrb_amodal --same_aug --hm_disturb 0.05 --lost_disturb 0.4 --fp_disturb 0.1 --kd_weight $i --gpus 0  --batch_size 16
    CUDA_VISIBLE_DEVICES=0 python test.py tracking --exp_id mot17_half-$i --dataset mot --dataset_version 17halfval --pre_hm --ltrb_amodal --track_thresh 0.4 --pre_thresh 0.5 --load_model ../exp/tracking/mot17_half-$i/model_last.pth --batch_size 1

done


