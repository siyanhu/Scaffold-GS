scene='scene_stairs/train_full_byorder_85'

main_sample_rate=1 # sample 1 image per every _ images.
exp_name='output_lod0_20241002'

voxel_size=0.001
update_init_factor=4
appearance_dim=0
ratio=1

./train.sh -d ${scene} \
-l ${exp_name} \
--gpu 0 --voxel_size ${voxel_size} --update_init_factor ${update_init_factor} \
--appearance_dim ${appearance_dim} --ratio ${ratio} --sample_rate ${main_sample_rate}