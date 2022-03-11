import os
import shutil
from pathlib import Path

data_dir = 'datasets/shapenet/ShapeNetV1processed'
camera_dir = 'datasets/shapenet/ShapeNetRendering'

datasets = {
    '02828884': 'bench', '03001627': 'chair', 
    '04379243': 'table', '02933112': 'cabinet', 
    '04256520': 'sofa'
}

def copy_data(dest='final_eval_data', num_data=1):
    if not Path(dest).is_dir():
        os.mkdir(dest)

    for k in datasets.keys():
        f = os.path.join(dest, datasets[k])
        
        if not Path(f).is_dir():
            print(f'Creating main folder {f}')
            os.mkdir(f)

        dd = os.path.join(data_dir, k)
        cd = os.path.join(camera_dir, k)
        obj_list = os.listdir(dd)

        for i in range(num_data):
            dest_obj = os.path.join(dest, datasets[k], str(i))

            if i >= len(obj_list):
                continue

            inside_k = os.path.join(dd, obj_list[i])

            if not Path(dest_obj).is_dir():
                print(f' - Creating subfolder {dest_obj}')
                os.mkdir(dest_obj)

            obj_img_dir = os.path.join(inside_k, 'images')
            mesh_dir = os.path.join(inside_k, 'mesh.pt')
            metadata_dr = os.path.join(inside_k, 'metadata.pt')
            obj_img_list = sorted(os.listdir(obj_img_dir))

            for j in range(2):
                img_to_copy = os.path.join(obj_img_dir, obj_img_list[j])
                img_dest = os.path.join(dest_obj, datasets[k] + str(j) + '.png')
                
                if not Path(img_dest).is_file():
                    shutil.copy(img_to_copy, img_dest)
                    # print(f' - Copying image: {img_dest}')

            mesh_dest = os.path.join(dest_obj, 'mesh.pt')
            metadata_dest = os.path.join(dest_obj, 'metadata.pt')
            
            if not Path(mesh_dest).is_file():
                shutil.copy(mesh_dir, mesh_dest)
                # print(f' - Copying mesh.pt: {mesh_dest}\n')

            if not Path(metadata_dest).is_file():
                shutil.copy(metadata_dr, metadata_dest)
                # print(f' - Copying metadata.pt: {metadata_dest}\n')

            rendering_md_dir = os.path.join(cd, obj_list[i], 'rendering', 'rendering_metadata.txt')
            camera_dest = os.path.join(dest_obj, 'cameras.txt')

            if not Path(camera_dest).is_file():
                shutil.copy(rendering_md_dir, camera_dest)
                # print(f' - Copying cameras.txt: {camera_dest}\n')

if __name__ == '__main__':
    copy_data(num_data=50)