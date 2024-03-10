voxel_size=0.001
update_init_factor=4
appearance_dim=0
ratio=1

./train.sh -d scene_stairs/train_full_byorder_85 \
-l output_sparse_setting_sc0 \
--gpu 0 --voxel_size ${voxel_size} --update_init_factor ${update_init_factor} \
--appearance_dim ${appearance_dim} --ratio ${ratio}