scene='scene_chess/test_full_byorder_59'
model='scene_chess/train_full_byorder_85'
exp_name='output_lod0'
# exp_ts='2024-04-22_12:35:03'
ratio=1

# echo "from " + "$PWD"/outputs/${model}/${exp_name}/${exp_ts}

# echo "to " + "$PWD"/data/${model}/${exp_name}

# cp -r "$PWD"/outputs/${model}/${exp_name}/${exp_ts} "$PWD"/data/${model}/${exp_ts}

# mv "$PWD"/data/${model}/${exp_ts} "$PWD"/data/${model}/${exp_name}

# rm -rf data/${model}/${exp_name}/test

python evaluate_dof.py -s data/${scene} -m data/${model}/${exp_name}

# python evaluate_dof.py -s data/${model} -m data/${model}/${exp_name}