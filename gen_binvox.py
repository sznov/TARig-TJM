# Process dataset so that .binvox files are generated for each .obj file.
import argparse
import os
import glob
import shutil

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-folder', type=str, required=True)
    args = parser.parse_args()
    dataset_dir = args.dataset_folder
    obj_dir = os.path.join(dataset_dir, "obj_remesh")

    for obj_filename in glob.glob(os.path.join(obj_dir, "*.obj")):
        print(obj_filename)
        os.system("binvox.exe -d 88 " + obj_filename)
        initial_binvox_filepath = obj_filename.replace(".obj", ".binvox") 
        
        # move binvox file to "vox" folder
        target_dir = os.path.join(dataset_dir, "vox")
        target_binvox_filepath = os.path.join(target_dir, os.path.basename(initial_binvox_filepath.replace("_remesh.binvox", ".binvox")))
        os.makedirs(target_dir, exist_ok=True)
        shutil.move(initial_binvox_filepath, target_binvox_filepath)