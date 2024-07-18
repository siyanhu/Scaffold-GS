import numpy as np
import root_file_io as fio


def cate_new_sfmPaths(pathList):
    train_sfm_dict = {}
    test_sfm_dict = {}
    for sfm_pth in pathList:
        (sp_dir, sp_name, sp_ext) = fio.get_filename_components(sfm_pth)
        combo = sp_name.split('_')
        if len(combo) < 2:
            continue
        scene_tag = 'scene_' + combo[0]
        func_tag = combo[1]
        if func_tag == 'test':
            test_sfm_dict[scene_tag] = sfm_pth
        if func_tag == 'train':
            train_sfm_dict[scene_tag] = sfm_pth
    return train_sfm_dict, test_sfm_dict



def read_sfm_file(sfm_pth):
    sfm_content = {}
    with open(sfm_pth, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                image_name = elems[0]
                qvec = elems[1:5]
                tvec = elems[5:8]
                new_fx = elems[-1]
                sfm_content[image_name] = {'image_name': image_name, 'qvec': qvec, 'tvec':tvec, 'fx': new_fx}
    return sfm_content


def merge_sfm_to_sparse(sparse_intr_path, sparse_extr_path, sfm_content):
    sparse_intr_backup_path = sparse_intr_path.replace('cameras.txt', 'cameras_dslam.txt')
    sparse_extr_backup_path = sparse_extr_path.replace('images.txt', 'images_dslam.txt')

    new_extr_lines = []
    # image_name_cameraID = {}
    fxx = 0.0

    with open(sparse_extr_path, 'r') as fextr_r:
        while True:
            line = fextr_r.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                image_id = int(elems[0])
                qvec = np.array(tuple(map(float, elems[1:5])))
                tvec = np.array(tuple(map(float, elems[5:8])))
                camera_id = int(elems[8])
                image_name = elems[9]
                point3D_elems = fextr_r.readline().split()
                
                # if camera_id not in image_name_cameraID:
                #     image_name_cameraID[camera_id] = []
                # image_name_cameraID[camera_id].append(image_name)

                if image_name not in sfm_content:
                    continue
                new_sfm = sfm_content[image_name]
                new_qvec = new_sfm['qvec']
                new_tvec = new_sfm['tvec']
                fxx = new_sfm['fx']

                new_extr_combo = []
                new_extr_combo.append(str(image_id))
                new_extr_combo += new_qvec
                new_extr_combo += new_tvec
                new_extr_combo.append(str(camera_id))
                new_extr_combo.append(image_name)
                new_extr_lines.append(' '.join(new_extr_combo))
                new_extr_lines.append(' '.join(point3D_elems))
            elif line[0] == "#":
                new_extr_lines.append(line)
    
    fio.move_file(sparse_extr_path, sparse_extr_backup_path)
    with open(sparse_extr_path, 'w') as fextr_w:
        for line in new_extr_lines:
            fextr_w.write(f"{line}\n")

    # related_camera_ids = list(image_name_cameraID.keys())
    new_intr_lines = []
    with open(sparse_intr_path, 'r') as fintr_r:
        while True:
            line = fintr_r.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                camera_id = int(elems[0])
                model = elems[1]
                width = int(elems[2])
                height = int(elems[3])
                fx = elems[4]
                extra = elems[5:]
                new_model = [str(camera_id), model, str(width), str(height), str(fxx)]
                new_model += [str(ln) for ln in extra]
                new_intr_lines.append(' '.join(new_model))
            elif line[0] == "#":
                new_intr_lines.append(line)
                
    fio.move_file(sparse_intr_path, sparse_intr_backup_path)
    with open(sparse_intr_path, 'w') as fintr_w:
        for line in new_intr_lines:
            fintr_w.write(f"{line}\n")


if __name__=='__main__':
    scene_tags = ['scene_chess', 'scene_fire', 'scene_heads', 'scene_office', 'scene_pumpkin', 'scene_redkitchen', 'scene_stairs']
    data_dir = fio.createPath(fio.sep, [fio.getParentDir(), 'data'])

    sfm_extr_dir = fio.createPath(fio.sep, [data_dir, 'sfm_ext'])
    sfm_file_pathsList = fio.traverse_dir(sfm_extr_dir, full_path=True, towards_sub=False)
    train_sfm_pathDict, test_sfm_pathDict = cate_new_sfmPaths(sfm_file_pathsList)

    for stag in scene_tags:
        print(stag)
        scene_dir = fio.createPath(fio.sep, [data_dir, stag])

        train_extr_path = fio.createPath(fio.sep, [scene_dir, 'train_full_byorder_85', 'sparse', '0'], 'images.txt')
        train_intr_path = fio.createPath(fio.sep, [scene_dir, 'train_full_byorder_85', 'sparse', '0'], 'cameras.txt')
        
        train_sfm_path = train_sfm_pathDict[stag]
        train_sfm_content = read_sfm_file(train_sfm_path)
        merge_sfm_to_sparse(train_intr_path, train_extr_path, train_sfm_content)


        test_extr_path = fio.createPath(fio.sep, [scene_dir, 'test_full_byorder_59', 'sparse', '0'], 'images.txt')
        test_intr_path = fio.createPath(fio.sep, [scene_dir, 'test_full_byorder_59', 'sparse', '0'], 'cameras.txt')
        test_sfm_path = test_sfm_pathDict[stag]
        test_sfm_content = read_sfm_file(test_sfm_path)
        merge_sfm_to_sparse(test_intr_path, test_extr_path, test_sfm_content)
