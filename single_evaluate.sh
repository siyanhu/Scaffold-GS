scene='scene_fire/test_full_byorder_59'
model='scene_fire/train_full_byorder_85'

main_sample_rate=1  # sample 1 image per every _ images.
exp_name='output_lod0_deblur_90pnt'
# exp_name='output_lod0'

exp_ts=''
ratio=1

# echo "from " + "$PWD"/outputs/${model}/${exp_name}/${exp_ts}

# echo "to " + "$PWD"/data/${model}/${exp_name}

# cp -r "$PWD"/outputs/${model}/${exp_name}/${exp_ts} "$PWD"/data/${model}/${exp_ts}

# mv "$PWD"/data/${model}/${exp_ts} "$PWD"/data/${model}/${exp_name}

# rm -rf data/${model}/${exp_name}/test

python evaluate_dof.py -s data/${scene} -m data/${model}/${exp_name} --sample_rate ${main_sample_rate} -r 1

# python evaluate_dof.py -s data/${model} -m data/${model}/${exp_name}
