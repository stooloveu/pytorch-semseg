python  -W ignore test_icnet_instance_seg.py \
 --model_path runs/icnet_cityscapes_instance_segmentation_resume/99609/icnet_is_cityscapes_instance_segmentation_08000_model.pkl \
 --dataset cityscapes_instance_segmentation \
 --img_path datasets/cityscapes/leftImg8bit/train/ulm/ulm_000010_000019_leftImg8bit.png \
 --out_path out.png \
 --use_original_icnet \
 --use_gt_sem_map


# many people 11
# --img_path datasets/cityscapes/leftImg8bit/train/cologne/cologne_000046_000019_leftImg8bit.png \
# --img_path datasets/cityscapes/leftImg8bit/train/stuttgart/stuttgart_000030_000019_leftImg8bit.png \

# many cars 13
# --img_path datasets/cityscapes/leftImg8bit/train/cologne/cologne_000010_000019_leftImg8bit.png \
# --img_path datasets/cityscapes/leftImg8bit/test/leverkusen/leverkusen_000012_000019_leftImg8bit.png \

# many motorcycles 17
# --img_path datasets/cityscapes/leftImg8bit/train/ulm/ulm_000010_000019_leftImg8bit.png

# 0.001 lr for 1000 iter, inter-class loss, then 0.001 for another 1000 iter
# --model_path runs/icnet_cityscapes_instance_segmentation_resume/92175/icnet_is_cityscapes_instance_segmentation_best_model.pkl \

# poly lr, from above, trained 5000 iter
# --model_path runs/icnet_cityscapes_instance_segmentation_resume/44241/icnet_is_cityscapes_instance_segmentation_05000_model.pkl \

# corrected loss, 1.0, 1.0, 0.001 and 0.5/2.0 from aove, trained for 500 iter
# --model_path runs/icnet_cityscapes_instance_segmentation_resume/31866/icnet_is_cityscapes_instance_segmentation_00500_model.pkl \

# corrected loss, from scratch, 5.0, 1.0, 0.001 and 0.1/1.5 from scratch, trained for 3000 iter
# --model_path runs/icnet_cityscapes_instance_segmentation/57365/icnet_is_cityscapes_instance_segmentation_03000_model.pkl \

# corrected loss, from scratch, 1.0, 1.0, 0.001 and 0.3/1.5 from above, trained for 5000 iter
# --model_path runs/icnet_cityscapes_instance_segmentation_resume/30207/icnet_is_cityscapes_instance_segmentation_05000_model.pkl \

# corrected loss, with delta_var, from scratch, 1.0, 1.0, 0.1 and 0.25/1.0/6.0 from scratch, 0.01 init lr, trained for 5000 iter
# --model_path runs/icnet_cityscapes_instance_segmentation/76082/icnet_is_cityscapes_instance_segmentation_05000_model.pkl \


# corrected loss, with delta_var, from scratch, 1.0, 1.0, 0.1 and 0.25/1.0/6.0 from above, 0.004 init lr, trained for 7000 iter, loss 0.39
# --model_path runs/icnet_cityscapes_instance_segmentation_resume/72870/icnet_is_cityscapes_instance_segmentation_07000_model.pkl \

# corrected loss, with delta_var, from scratch, 1.0, 1.0, 0.1 and 0.25/1.0/6.0 from above, 0.003 init lr, trained for 6000 iter, loss 0.34
# --model_path runs/icnet_cityscapes_instance_segmentation_resume/86262/icnet_is_cityscapes_instance_segmentation_06000_model.pkl \

# corrected loss, with delta_var, from scratch, 1.0, 1.0, 0.1 and 0.25/1.0/6.0 from above, 0.004 init lr, trained for 4000 iter, loss 0.30
# --model_path runs/icnet_cityscapes_instance_segmentation_resume/76996/icnet_is_cityscapes_instance_segmentation_04000_model.pkl \

# corrected loss, with delta_var, 1.0, 1.0, 0.1 and 0.25/1.0/6.0 from above, 0.003 init lr, adadelta, trained for 500 iter, loss 0.28
# --model_path runs/icnet_cityscapes_instance_segmentation_resume/72369/icnet_is_cityscapes_instance_segmentation_00500_model.pkl \

# corrected loss, with delta_var, 1.0, 1.0, 0.1 and 0.25/1.0/6.0 from above, 0.00003 init lr, adamax, trained for 3000 iter, loss 0.27
# --model_path runs/icnet_cityscapes_instance_segmentation_resume/29641/icnet_is_cityscapes_instance_segmentation_03000_model.pkl \

# corrected loss, with delta_var, 1.0, 1.0, 0.1 and 0.25/1.0/6.0 from above, 0.00006 init lr, 0.00002 end lr, adamax, trained for 8000 iter, loss 0.25
# --model_path runs/icnet_cityscapes_instance_segmentation_resume/50965/icnet_is_cityscapes_instance_segmentation_08000_model.pkl \

# corrected loss, with delta_var, 1.0, 1.0, 0.1 and 0.25/1.0/6.0 from above, 0.00002 init lr, 0.00002 end lr, adamax, trained for 3500 iter, loss 0.24
# --model_path runs/icnet_cityscapes_instance_segmentation_resume/9056/icnet_is_cityscapes_instance_segmentation_03500_model.pkl \

# corrected loss, with delta_var, 1.0, 1.0, 0.1 and 0.25/1.0/6.0 from above, 0.00006 init lr, 0.00002 end lr, adamax, trained for 8000 iter, loss 0.24
# --model_path runs/icnet_cityscapes_instance_segmentation_resume/99609/icnet_is_cityscapes_instance_segmentation_08000_model.pkl \

# --img_path datasets/cityscapes/leftImg8bit/test/berlin/berlin_000000_000019_leftImg8bit.png \