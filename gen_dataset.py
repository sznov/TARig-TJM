#-------------------------------------------------------------------------------
# Name:        gen_dataset.py
# Purpose:     Script to generate data for skeleton and connectivity predition stage
#              Change dataset_folder to the folder where you put the downloaded pre-processed data
# RigNet Copyright 2020 University of Massachusetts
# RigNet is made available under General Public License Version 3 (GPLv3), or under a Commercial License.
# Please see the LICENSE README.txt file in the main directory for more information and instruction on using and licensing RigNet.
#-------------------------------------------------------------------------------

import argparse
import os
import shutil
import numpy as np 
import open3d as o3d
from multiprocessing import Pool
from utils.io_utils import mkdir_p
from utils.rig_parser import Info
from geometric_proc.common_ops import calc_surface_geodesic

DEFAULT_NUM_GEODESIC_NEIGHBORS = 10
DEFAULT_GEODESIC_BALL_RAIDUS = 0.06

def get_tpl_edges(remesh_obj_v, remesh_obj_f):
    edge_index = []
    for v in range(len(remesh_obj_v)):
        face_ids = np.argwhere(remesh_obj_f == v)[:, 0]
        neighbor_ids = []
        for face_id in face_ids:
            for v_id in range(3):
                if remesh_obj_f[face_id, v_id] != v:
                    neighbor_ids.append(remesh_obj_f[face_id, v_id])
        neighbor_ids = list(set(neighbor_ids))
        neighbor_ids = [np.array([v, n])[np.newaxis, :] for n in neighbor_ids]
        neighbor_ids = np.concatenate(neighbor_ids, axis=0)
        edge_index.append(neighbor_ids)
    edge_index = np.concatenate(edge_index, axis=0)
    return edge_index


def get_geo_edges(surface_geodesic, remesh_obj_v, num_geodesic_neighbors=DEFAULT_NUM_GEODESIC_NEIGHBORS, geodesic_ball_radius=DEFAULT_GEODESIC_BALL_RAIDUS):
    edge_index = []
    surface_geodesic += 1.0 * np.eye(len(surface_geodesic))  # remove self-loop edge here
    for i in range(len(remesh_obj_v)):
        geodesic_ball_samples = np.argwhere(surface_geodesic[i, :] <= geodesic_ball_radius).squeeze(1)
        if len(geodesic_ball_samples) > num_geodesic_neighbors:
            geodesic_ball_samples = np.random.choice(geodesic_ball_samples, num_geodesic_neighbors, replace=False)
        edge_index.append(np.concatenate((np.repeat(i, len(geodesic_ball_samples))[:, np.newaxis],
                                          geodesic_ball_samples[:, np.newaxis]), axis=1))
    edge_index = np.concatenate(edge_index, axis=0)
    return edge_index


def genDataset(split_name, num_geodesic_neighbors=DEFAULT_NUM_GEODESIC_NEIGHBORS, geodesic_ball_radius=DEFAULT_GEODESIC_BALL_RAIDUS):
    global dataset_folder

    model_list = []
    with open(os.path.join(dataset_folder, f'{split_name}_final.txt'), 'r') as f:
        for line in f:
            model_list.append(line.strip())

    mkdir_p(os.path.join(dataset_folder, split_name))

    # make sure all models have the same joint name order
    common_joint_name_list = []

    for model_id in model_list:
        # Skip if outputs already exist
        non_existent_output = False
        for output in ['_v.txt', '_tpl_e.txt', '_geo_e.txt', '_j.txt', '_adj.txt', '_attn_per_joint.txt', '.binvox']:
            if not os.path.exists(os.path.join(dataset_folder, f'{split_name}/{model_id}{output}')):
                non_existent_output = True

        if not non_existent_output:
            print(f'Skipping {model_id} because outputs already exist')
            continue

        remeshed_obj_filename = os.path.join(dataset_folder, f'obj_remesh/{model_id}_remesh.obj')
        info_filename = os.path.join(dataset_folder, f'rig_info/{model_id}.txt')
        remeshed_obj = o3d.io.read_triangle_mesh(remeshed_obj_filename)
        remesh_obj_v = np.asarray(remeshed_obj.vertices)
        if not remeshed_obj.has_vertex_normals():
            remeshed_obj.compute_vertex_normals()
        remesh_obj_vn = np.asarray(remeshed_obj.vertex_normals)
        remesh_obj_f = np.asarray(remeshed_obj.triangles)
        rig_info = Info(info_filename)

        # vertices
        vert_filename = os.path.join(dataset_folder, f'{split_name}/{model_id}_v.txt')
        input_feature = np.concatenate((remesh_obj_v, remesh_obj_vn), axis=1)
        np.savetxt(vert_filename, input_feature, fmt='%.6f')

        # topological edges
        edge_index = get_tpl_edges(remesh_obj_v, remesh_obj_f)
        graph_filename = os.path.join(dataset_folder, f'{split_name}/{model_id}_tpl_e.txt')
        np.savetxt(graph_filename, edge_index, fmt='%d')

        # geodesic edges
        surface_geodesic = calc_surface_geodesic(remeshed_obj)
        edge_index = get_geo_edges(surface_geodesic, remesh_obj_v, num_geodesic_neighbors, geodesic_ball_radius)
        graph_filename = os.path.join(dataset_folder, f'{split_name}/{model_id}_geo_e.txt')
        np.savetxt(graph_filename, edge_index, fmt='%d')

        # joints
        joint_pos = rig_info.get_joint_dict()
        joint_name_list = list(joint_pos.keys())
        if len(common_joint_name_list) == 0:
            common_joint_name_list = joint_name_list
        else:
            assert common_joint_name_list == joint_name_list
        joint_pos_list = list(joint_pos.values())
        joint_pos_list = [np.array(i) for i in joint_pos_list]
        adjacent_matrix = rig_info.adjacent_matrix()
        joint_filename = os.path.join(dataset_folder, f'{split_name}/{model_id}_j.txt')
        adj_filename = os.path.join(dataset_folder, f'{split_name}/{model_id}_adj.txt')
        np.savetxt(adj_filename, adjacent_matrix, fmt='%d')
        np.savetxt(joint_filename, np.array(joint_pos_list), fmt='%.6f')

        # pre_trained attn
        shutil.copyfile(os.path.join(dataset_folder, f'pretrain_attention/{model_id}_attn_per_joint.txt'), 
                        os.path.join(dataset_folder, f'{split_name}/{model_id}_attn_per_joint.txt'))
        
        # voxel
        shutil.copyfile(os.path.join(dataset_folder, f'vox/{model_id}.binvox'), 
                        os.path.join(dataset_folder, f'{split_name}/{model_id}.binvox'))
   
    joint_names_filename = os.path.join(dataset_folder, f'ordered_joint_names.txt')
    if not os.path.exists(joint_names_filename):
        with open(joint_names_filename, 'w') as f:
            for item in common_joint_name_list:
                f.write("%s\n" % item)
    else:
        with open(joint_names_filename, 'r') as f:
            existing_common_joint_name_list = [line.strip() for line in f]
        assert existing_common_joint_name_list == common_joint_name_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-folder', type=str, required=True)
    parser.add_argument('--num-geodesic-neighbors', type=int, default=DEFAULT_NUM_GEODESIC_NEIGHBORS)
    parser.add_argument('--geodesic-ball-radius', type=float, default=DEFAULT_GEODESIC_BALL_RAIDUS)
    args = parser.parse_args()

    dataset_folder = args.dataset_folder
    genDataset('train', args.num_geodesic_neighbors, args.geodesic_ball_radius)
    genDataset('val', args.num_geodesic_neighbors, args.geodesic_ball_radius)
    genDataset('test', args.num_geodesic_neighbors, args.geodesic_ball_radius)
