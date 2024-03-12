scene='scene_stairs/test_full_byorder_59'
model='scene_stairs/train_full_byorder_85'
exp_name='output_sparse_setting_sc0'
exp_ts='2024-03-10_22\:02\:10'
ratio=1

# ./evaluate.sh -d ${scene} \
# -m ${model} \
# -l ${exp_name/exp_ts} \
# --gpu 0 \
# --r ${ratio}

# cp -r outputs/scene_stairs/train_full_byorder_85/output_sparse_setting_sc0/2024-03-10_22\:02\:10 \
# data/scene_stairs/train_full_byorder_85/output_sparse_setting_sc0

# python evaluate.py -s data/scene_stairs/test_full_byorder_59 -m outputs/scene_stairs/train_full_byorder_85/output_sparse_setting_sc0/2024-03-10_22\:02\:10
python evaluate_dof.py -s data/scene_stairs/test_full_byorder_59 -m data/scene_stairs/train_full_byorder_85/current
# python evaluate_dof.py -s data/scene_stairs/train_full_byorder_85 -m data/scene_stairs/train_full_byorder_85/current