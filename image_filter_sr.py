import cv2
from cv2 import dnn_superres
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


def super_resolution(image_path):
    (imdir, imname, imext) = fio.get_filename_components(image_path)
    img = cv2.imread(image_path)
    trained_model_path = fio.createPath(fio.sep, [fio.getParentDir(), 'sr_models'], 'EDSR_x2.pb')
    if fio.file_exist(trained_model_path) == False:
        return
    sr = dnn_superres.DnnSuperResImpl_create()
    sr.readModel(trained_model_path)
    sr.setModel('edsr', 4)
    result = sr.upsample(img)

    new_dir = fio.createPath(fio.sep, [imdir + '_x4'])
    fio.ensure_dir(new_dir)
    new_path = fio.createPath(fio.sep, [new_dir], imname + '.' + imext)
    cv2.imwrite(new_path, result)



if __name__=='__main__':
    data_dir = fio.createPath(fio.sep, [fio.getParentDir(), 'data'])
    scene_tags = ['scene_redkitchen']
    
    for stag in scene_tags:
        scene_dir = fio.createPath(fio.sep, [data_dir, stag, 'train_full_byorder_85', 'images'])
        seq_dir_list = fio.traverse_dir(scene_dir, full_path=True, towards_sub=False)
        seq_dir_list = fio.filter_folder(seq_dir_list, filter_out=False, filter_text='seq')

        llp_trend = {}
        for seq_pth in seq_dir_list:
            seq_dir = ''
            (seq_dir, seq_name, seq_ext) = fio.get_filename_components(seq_pth)
            image_paths = fio.traverse_dir(seq_pth, full_path=True, towards_sub=False)
            for img_pth in image_paths:
                [img_dir, img_name, img_ext] = fio.get_filename_components(img_pth)
                # llp = calculate_laplacian(img_pth)
                # if llp < 60.0:
                super_resolution(img_pth)
