#-------------------------------------------------------------------------------
# Name:        compute_surface_geodesic.py
# Purpose:     This script calculates surface geodesic distance between all pair of vertices.
#              It doesn't rely on mesh topology. Surface is densely sampled to form a graph and get shortest path between samples.
#              Geodesic distance between pair of vertices are the geodesic distance between nearest samples to both vertices.
# RigNet Copyright 2020 University of Massachusetts
# RigNet is made available under General Public License Version 3 (GPLv3), or under a Commercial License.
# Please see the LICENSE README.txt file in the main directory for more information and instruction on using and licensing RigNet.
#-------------------------------------------------------------------------------

import sys
sys.path.append("./")
import os
import glob
import numpy as np
import open3d as o3d
from geometric_proc.common_ops import calc_surface_geodesic


DATASET_FOLDER = "dataset/" # TODO Specify

OBJ_FOLDER        = os.path.join(DATASET_FOLDER, "obj/")
REMESH_OBJ_FOLDER = os.path.join(DATASET_FOLDER, "obj_remesh/")
RESULTS_FOLDER    = os.path.join(DATASET_FOLDER, "surface_geodesic/")

def main():
    target_folder = REMESH_OBJ_FOLDER
    results_folder = RESULTS_FOLDER
    target_folder = glob.glob(target_folder + '*.obj')
    for obj_filename in target_folder:
        model_id = os.path.splitext(os.path.basename(obj_filename))[0].replace("_remesh", "")
        print(model_id)
        surface_geodesic = calc_surface_geodesic(o3d.io.read_triangle_mesh(obj_filename))
        os.makedirs(results_folder, exist_ok=True)
        np.save(os.path.join(results_folder, "{:s}_surface_geo.npy".format(model_id)), surface_geodesic.astype(np.float16))


if __name__ == '__main__':
    main()
