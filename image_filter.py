import cv2
import numpy as np
import scipy.stats as stats
import pylab as pl

import root_file_io as fio


def calculate_laplacian(image_path):
    if fio.file_exist(image_path) == False:
        return -1
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    laplacian_var = cv2.Laplacian(image, cv2.CV_64F).var()
    return laplacian_var


def save_distribution(h, scene_dir):
    fit = stats.norm.pdf(h, np.mean(h), np.std(h))  #this is a fitting indeed
    pl.plot(h,fit,'-o')
    pl.hist(h)
    save_path = fio.createPath(fio.sep, [scene_dir], 'llp_dist.png')
    pl.savefig(save_path)


def filter_keys_by_value(d, threshold):
    return [str(key) for key, value in d.items() if value > threshold]


def save_keep_tags(save_tag_path, left_tags):
    with open(save_tag_path, 'w') as f1:
        for line in left_tags:
            f1.write(f"{line}\n")


def read_keep_tags(save_tag_path):
    left_tags = []
    with open(save_tag_path, 'r') as f1:
        rslt = f1.readlines()
        left_tags = [x.strip() for x in rslt]
    return left_tags



def filter_extrinsics_text(original_Path, left_tags):
    save_lines = []
    with open(original_Path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                image_name = elems[9]
                image_name = image_name.replace('.color.png', '.png')
                if image_name in left_tags:
                    save_lines.append(line)
                    nextline = fid.readline()
                    nextline = nextline.strip()
                    save_lines.append(nextline)
            else:
                save_lines.append(line.strip())

    (exdir, exname, exext) = fio.get_filename_components(original_Path)
    new_path_for_full = fio.createPath(fio.sep, [exdir], 'images_full.txt')
    fio.move_file(original_Path, new_path_for_full)
    if fio.file_exist(original_Path):
        fio.move_file(original_Path, '/data/test_images.txt')
    with open(original_Path, 'w') as f:
        for line in save_lines:
            f.write(f"{line}\n")
    

if __name__=='__main__':
    data_dir = fio.createPath(fio.sep, [fio.getParentDir(), 'data'])
    scene_tags = ['scene_fire', 'scene_pumpkin']
    
    for stag in scene_tags:
        scene_dir = fio.createPath(fio.sep, [data_dir, stag, 'train_full_byorder_85', 'images'])
        seq_dir_list = fio.traverse_dir(scene_dir, full_path=True, towards_sub=False)
        seq_dir_list = fio.filter_folder(seq_dir_list, filter_out=False, filter_text='seq')

        save_path_lt = fio.createPath(fio.sep, [data_dir, stag, 'train_full_byorder_85', 'sparse', '0'], 'left_90_tags.txt')
        left_image_tags = []
        if fio.file_exist(save_path_lt):
            left_image_tags = read_keep_tags(save_path_lt)
        else:
            llp_trend = {}
            for seq_pth in seq_dir_list:
                seq_dir = ''
                (seq_dir, seq_name, seq_ext) = fio.get_filename_components(seq_pth)
                image_paths = fio.traverse_dir(seq_pth, full_path=True, towards_sub=False)
                for img_pth in image_paths:
                    [img_dir, img_name, img_ext] = fio.get_filename_components(img_pth)
                    llp = calculate_laplacian(img_pth)
                    key = fio.sep.join([seq_name, img_name]) + '.' + img_ext
                    llp_trend[key] = llp
            llp_distribution = llp_trend.values()
            h = sorted(llp_distribution)
            left = np.percentile(h, [10, 15, 25, 50, 75, 85, 90])
            target = left[0]
            left_image_tags = filter_keys_by_value(llp_trend, target)
            save_keep_tags(save_path_lt, left_image_tags)
            print("saved only ",  str(100 - target), "percent samples, now has", 
                  str(len(left_image_tags)), "from totally ", str(len(llp_distribution)))
        extrinsic_filepath = fio.createPath(fio.sep, [data_dir, stag, 'train_full_byorder_85', 'sparse', '0'], 'images.txt')
        if fio.file_exist(extrinsic_filepath) == False:
            continue
        filter_extrinsics_text(extrinsic_filepath, left_image_tags)