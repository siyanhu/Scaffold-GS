scene='scene_GreatCourt_bright/test_full_byorder_59'
model='scene_GreatCourt_bright/train_full_byorder_85'
exp_name='output_lod0_seq3'
exp_ts='2024-05-06_18:57:24'
ratio=1

echo "from " + "$PWD"/outputs/${model}/${exp_name}/${exp_ts}

echo "to " + "$PWD"/data/${model}/${exp_name}

cp -r "$PWD"/outputs/${model}/${exp_name}/${exp_ts} "$PWD"/data/${model}/${exp_ts}

mv "$PWD"/data/${model}/${exp_ts} "$PWD"/data/${model}/${exp_name}

rm -rf data/${model}/${exp_name}/test

python evaluate_dof.py -s data/${scene} -m data/${model}/${exp_name}

# python evaluate_dof.py -s data/${model} -m data/${model}/${exp_name}