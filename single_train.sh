scene='scene_ShopFacade/train_full_patch_85'

main_sample_rate=2  # sample 1 image per every _ images.
exp_name='output_train_exp'${main_sample_rate}

voxel_size=0.001
update_init_factor=4
appearance_dim=0
ratio=1

./train.sh -d ${scene} \
-l ${exp_name} \
--gpu 0 --voxel_size ${voxel_size} --update_init_factor ${update_init_factor} \
--appearance_dim ${appearance_dim} --ratio ${ratio} --sample_rate ${main_sample_rate}