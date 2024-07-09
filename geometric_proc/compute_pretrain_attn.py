#-------------------------------------------------------------------------------
# Name:        compute_pretrain_attn.py
# Purpose:     This script prepares the supervision for attention module pretraining.
#              It shoots rays from each joint around the plane perpendicular to the bone, which hit the surface.
#              Vertices near the valid hits are marked as 1, otherwise 0.
# RigNet Copyright 2020 University of Massachusetts
# RigNet is made available under General Public License Version 3 (GPLv3), or under a Commercial License.
# Please see the LICENSE README.txt file in the main directory for more information and instruction on using and licensing RigNet.
#-------------------------------------------------------------------------------

import sys
sys.path.append("./")

import argparse
import copy
import os
import numpy as np
import open3d as o3d
import trimesh
from utils.rig_parser import Info

def get_perpend_vec(v):
    max_dim = np.argmax(np.abs(v))
    if max_dim == 0:
        u_0 = np.array([(-2.0 * v[1] - 1.0 * v[2]) / (v[0]+1e-10), 2.0, 1.0])
    elif max_dim == 1:
        u_0 = np.array([1.0, (-1.0 * v[0] - 2.0 * v[2]) / (v[1]+1e-10), 2.0])
    elif max_dim == 2:
        u_0 = np.array([1.0, 2.0, (-1.0 * v[0] - 2.0 * v[1]) / (v[2]+1e-10)])
    u_0 /= np.linalg.norm(u_0)
    return u_0


def cal_perpendicular_dir(p_pos, ch_pos):
    global ray_per_sample
    dirs = []
    v = (ch_pos - p_pos).squeeze()
    v = v / (np.linalg.norm(v)+1e-10)
    u_0 = get_perpend_vec(v)
    w = np.cross(v, u_0)
    w = w / (np.linalg.norm(w)+1e-10)
    for angle in np.arange(0, 2*np.pi, 2*np.pi/ray_per_sample):
        u = np.cos(angle) * u_0 + np.sin(angle) * w
        u = u / (np.linalg.norm(u)+1e-10)
        dirs.append(u[np.newaxis, :])
    dirs = np.concatenate(dirs, axis=0)
    return dirs


def form_rays(skel):
    '''
    generate rays from joints, with perpendicular direction
    :param skel: input skeleton
    :return: ray origins and ray directions
    '''
    origins = []
    dirs = []
    joint_names = []
    this_level = [skel.root]
    while this_level:
        next_level = []
        for p_node in this_level:
            next_level += p_node.children
            p_pos = np.array(p_node.pos)[np.newaxis, :]
            for c_node in p_node.children:
                c_pos = np.array(c_node.pos)[np.newaxis, :]
                origin_bone = np.concatenate(([p_pos, c_pos]), axis=0)
                dir_bone = cal_perpendicular_dir(p_pos, c_pos)
                origins.append(np.repeat(origin_bone, len(dir_bone), axis=0))
                dirs.append(np.tile(dir_bone, (len(origin_bone), 1)))
                joint_names.append([p_node.name])
                joint_names.append([c_node.name])
        this_level = next_level
    origins = np.concatenate(origins, axis=0)
    dirs = np.concatenate(dirs, axis=0)
    joint_names = np.concatenate(joint_names, axis=0)
    return origins, dirs, joint_names


def shoot_rays(mesh, origins, ray_dir, debug=False, dataset_folder=None, model_id=None):
    '''
    shoot rays and record the first hit distance, as well as all vertices on the hit faces.
    :param mesh: input mesh (trimesh)
    :param origins: origin of rays
    :param ray_dir: direction of rays
    :return: all vertices indices on the hit face, the distance of first hit for each ray.
    '''
    global ray_per_sample
    RayMeshIntersector = trimesh.ray.ray_triangle.RayMeshIntersector(mesh)
    locations, index_ray, index_tri = RayMeshIntersector.intersects_location(origins, ray_dir + 1e-15)
    locations_per_ray = []
    index_tri_per_ray = []
    for i in range(len(ray_dir)):
        locations_per_ray.append(locations[index_ray == i])
        index_tri_per_ray.append(index_tri[index_ray == i])
    all_hit_pos = []
    all_hit_tri = []
    all_hit_ori = []
    all_hit_ori_id = []
    for ori_id in np.arange(0, len(ray_dir), ray_per_sample):
        hit_pos = []
        hit_tri = []
        hit_dist = []
        hit_ori_id = []
        for i in range(ray_per_sample):
            ray_id = int(ori_id + i)
            if len(locations_per_ray[ray_id]) > 1:
                closest_hit_id = np.argmin(np.linalg.norm(locations_per_ray[ray_id] - origins[ray_id], axis=1))
                hit_pos.append(locations_per_ray[ray_id][closest_hit_id][np.newaxis, :])
                hit_dist.append(np.linalg.norm(locations_per_ray[ray_id][closest_hit_id] - origins[ray_id]))
                hit_tri.append(index_tri_per_ray[ray_id][closest_hit_id])
                hit_ori_id.append(int(ori_id/ray_per_sample))
            elif len(locations_per_ray[ray_id]) == 1:
                hit_pos.append(locations_per_ray[ray_id])
                hit_dist.append(np.linalg.norm(locations_per_ray[ray_id][0] - origins[ray_id]))
                hit_tri.append(index_tri_per_ray[ray_id][0])
                hit_ori_id.append(int(ori_id/ray_per_sample))

        if len(hit_pos) == 0: # no hit, pick nearby faces
            hit_tri = trimesh.proximity.nearby_faces(mesh, origins[int(ori_id + 0)][np.newaxis, :])[0]
            hit_vertices = mesh.faces[hit_tri].flatten()
            hit_pos = [np.array(mesh.vertices[i])[np.newaxis, :] for i in hit_vertices]
            hit_dist = [np.linalg.norm(hit_pos[i].squeeze() - origins[int(ori_id + 0)]) for i in range(len(hit_pos))]
            hit_tri = np.repeat(hit_tri, 3)
            hit_ori_id = np.repeat(int(ori_id / ray_per_sample), len(hit_tri))

        hit_pos = np.concatenate(hit_pos, axis=0)
        hit_dist = np.array(hit_dist)
        hit_tri = np.array(hit_tri)
        hit_ori_id = np.array(hit_ori_id)
        valid_ids = np.argwhere(hit_dist < np.percentile(hit_dist, 20) * 2).squeeze(1)
        hit_pos = hit_pos[valid_ids]
        hit_dist = hit_dist[valid_ids]
        hit_tri = hit_tri[valid_ids]
        hit_ori_id = hit_ori_id[valid_ids]

        all_hit_pos.append(hit_pos)
        all_hit_tri.append(hit_tri)
        all_hit_ori_id.append(hit_ori_id)
        all_hit_ori.append(origins[int(ori_id + 0)][np.newaxis, :])

    all_hit_pos = np.concatenate(all_hit_pos, axis=0)
    all_hit_tri = np.concatenate(all_hit_tri)
    all_hit_ori_id = np.concatenate(all_hit_ori_id)
    all_hit_ori = np.concatenate(all_hit_ori, axis=0)

    if debug:
        if dataset_folder is None or model_id is None:
            raise ValueError('dataset_folder and model_id must be provided when debug=True')
        import open3d as o3d
        from utils.vis_utils import drawSphere, find_lines_from_tree
        mesh_filename = f'{dataset_folder}/obj_remesh/{model_id}_remesh.obj'
        skel = Info(f'{dataset_folder}/rig_info/{model_id}.txt')
        # show mesh
        mesh_o3d = o3d.io.read_triangle_mesh(mesh_filename)
        mesh_ls = o3d.geometry.LineSet.create_from_triangle_mesh(mesh_o3d)
        mesh_ls.colors = o3d.utility.Vector3dVector([[0.8, 0.8, 0.8] for i in range(len(mesh_ls.lines))])
        # show skeleton
        line_list_skel = []
        joint_pos_list = []
        find_lines_from_tree(skel.root, line_list_skel, joint_pos_list)
        line_set_skel = o3d.geometry.LineSet()
        line_set_skel.points = o3d.utility.Vector3dVector(joint_pos_list)
        line_set_skel.lines = o3d.utility.Vector2iVector(line_list_skel)
        colors = [[1.0, 0.0, 0.0] for i in range(len(line_list_skel))]
        line_set_skel.colors = o3d.utility.Vector3dVector(colors)
        # show ray
        dpts = np.concatenate((all_hit_ori, all_hit_pos), axis=0)
        dlines = o3d.geometry.LineSet()
        dlines.points = o3d.utility.Vector3dVector(dpts)
        dlines.lines = o3d.utility.Vector2iVector([[all_hit_ori_id[i], len(all_hit_ori) + i] for i in range(len(all_hit_ori_id))])
        colors = [[0.0, 0.0, 1.0] for i in range(len(all_hit_ori_id))]
        dlines.colors = o3d.utility.Vector3dVector(colors)
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        vis.add_geometry(dlines)
        vis.add_geometry(mesh_ls)
        vis.add_geometry(line_set_skel)
        this_level = skel.root.children
        while this_level:
            next_level = []
            for p_node in this_level:
                vis.add_geometry(drawSphere(p_node.pos, 0.007, color=[1.0, 0.0, 0.0]))  # [0.3, 0.1, 0.1]
                next_level += p_node.children
            this_level = next_level
        vis.run()
        vis.destroy_window()

    return all_hit_pos, all_hit_ori_id, all_hit_ori


def normalize_mesh_rig(mesh, rig):
    # normalize mesh
    mesh_v = np.asarray(mesh.vertices)
    dims = [max(mesh_v[:, 0]) - min(mesh_v[:, 0]),
            max(mesh_v[:, 1]) - min(mesh_v[:, 1]),
            max(mesh_v[:, 2]) - min(mesh_v[:, 2])]
    scale = 1.0 / max(dims)
    pivot = np.array([(min(mesh_v[:, 0]) + max(mesh_v[:, 0])) / 2, 
                      (min(mesh_v[:, 1]) + max(mesh_v[:, 1])) / 2,
                      (min(mesh_v[:, 2]) + max(mesh_v[:, 2])) / 2])
    mesh_v[:, 0] -= pivot[0]
    mesh_v[:, 1] -= pivot[1]
    mesh_v[:, 2] -= pivot[2]
    mesh_v *= scale
    mesh.vertices = o3d.utility.Vector3dVector(mesh_v)

    # normalize rig
    for k, v in rig.joint_pos.items():
        rig.joint_pos[k] -= pivot
        rig.joint_pos[k] *= scale
    this_level = [rig.root]
    while this_level:
        next_level = []
        for node in this_level:
            node.pos = (np.array(node.pos) - pivot) * scale
            node.pos = (node.pos[0], node.pos[1], node.pos[2])
            for ch in node.children:
                next_level.append(ch)
        this_level = next_level

    return mesh, rig

if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description='Prepare supervision for attention module pretraining')
    argparser.add_argument('--dataset-folder', type=str, required=True, help='Dataset folder')
    argparser.add_argument('--rays-per-sample', type=int, default=14, help='Number of rays shoot from each joint (14 in original RigNet repo)')
    argparser.add_argument('--mask-dist-threshold', type=float, default=2.5e-2, help='Distance threshold for marking vertices as 1 (2e-2 in original RigNet repo)')
    argparser.add_argument('--subsample', type=int, default=None, help='Subsample mesh to this number of vertices (3000 in original RigNet repo)')
    args = argparser.parse_args()
    dataset_folder = args.dataset_folder
    obj_folder        = os.path.join(dataset_folder, "obj")
    remesh_obj_folder = os.path.join(dataset_folder, "obj_remesh")
    rig_info_folder   = os.path.join(dataset_folder, "rig_info")
    results_folder    = os.path.join(dataset_folder, "pretrain_attention")

    os.makedirs(results_folder, exist_ok=True)

    subsampling = args.subsample
    ray_per_sample = args.rays_per_sample

    model_list = [os.path.splitext(entry)[0] for entry in os.listdir(obj_folder) if entry.endswith('.obj')]

    for model_id in model_list:
        print(model_id)
        mesh = o3d.io.read_triangle_mesh(os.path.join(remesh_obj_folder, f'{model_id}_remesh.obj'))

        rig_info = Info(os.path.join(rig_info_folder, f'{model_id}.txt'))
        mesh, rig_info = normalize_mesh_rig(mesh, rig_info)
        mesh_ori = copy.deepcopy(mesh)
        vtx_ori = np.asarray(mesh.vertices)

        if subsampling:
            mesh = mesh.simplify_quadric_decimation(subsampling)

        mesh_trimesh = trimesh.Trimesh(vertices=np.asarray(mesh.vertices), faces=np.asarray(mesh.triangles), process=False)
        trimesh.repair.fix_normals(mesh_trimesh)
        origins, dirs, joint_names = form_rays(rig_info)

        hit_pos, all_hit_ori_id, all_hit_ori = shoot_rays(mesh_trimesh, origins, dirs, debug=False, model_id=model_id, dataset_folder=dataset_folder)
        def calculate_distances(vtx_ori, hit_pos, batch_size):
            # Calculate distances between each hit position and each vertex
            #
            # Memory-inefficent implementation: 
            #   return np.sqrt(np.sum((vtx_ori[np.newaxis, ...] - hit_pos[:, np.newaxis, :])**2, axis=2))
            num_batches = (vtx_ori.shape[0] - 1) // batch_size + 1
            dist = np.empty((hit_pos.shape[0], vtx_ori.shape[0]))
            for i in range(num_batches):
                start = i * batch_size
                end = start + batch_size
                dist[:, start:end] = np.sqrt(np.sum((vtx_ori[start:end] - hit_pos[:, None])**2, axis=2))
            return dist
        
        # HACK This is a sanity check to make sure we don't have too many hits.
        if hit_pos.shape[0] > 140000:
            msg = f'WARNING: {model_id} has {hit_pos.shape[0]} hits, which is more than 140000. This may cause memory issues. Skipping...'
            print(msg)
            # append to log file
            with open(os.path.join(results_folder, 'log.txt'), 'a') as f:
                f.write(msg + '\n')
            continue

        dist = calculate_distances(vtx_ori, hit_pos, batch_size=128)

        attn = np.zeros(len(vtx_ori), bool)
        attn_per_joint = np.zeros((len(rig_info.joint_pos), len(vtx_ori)), bool)
        joint_type_to_idx = {joint_type: idx for idx, joint_type in enumerate(rig_info.joint_pos)}

        for joint_id in np.unique(all_hit_ori_id):
            joint_name = joint_names[joint_id]
            joint_idx = joint_type_to_idx[joint_name]
            # HACK Based on joint name, set appropriate distance threshold (smallest for fingers)
            joint_name_lower = joint_name.lower()
            mask = dist < args.mask_dist_threshold
            for finger_substr in ['thumb', 'index', 'middle', 'ring', 'pinky']:
                if finger_substr in joint_name_lower:
                    mask = dist < args.mask_dist_threshold / 5.75
                    break
            num_nn = np.sum(np.sum(mask[np.argwhere(all_hit_ori_id == joint_id).squeeze(), :], axis=0) > 0)
            if num_nn < 6:
                # too few nearest points
                id_sort = np.argsort(np.linalg.norm(vtx_ori - all_hit_ori[joint_id][np.newaxis, :], axis=1))
                attn[id_sort[0:6]] = True
                attn_per_joint[joint_idx][id_sort[0:6]] = True
            else:
                id_nn = np.argwhere(np.sum(mask[np.argwhere(all_hit_ori_id == joint_id).squeeze(), :], axis=0) > 0).squeeze(1)
                attn[id_nn] = True
                attn_per_joint[joint_idx][id_nn] = True

        # Visualization
        def visualize():
            for joint_id, pos in rig_info.joint_pos.items():
                vis = o3d.visualization.Visualizer()
                vis.create_window()
                mesh_ls = o3d.geometry.LineSet.create_from_triangle_mesh(mesh_ori)
                mesh_ls.colors = o3d.utility.Vector3dVector([[0.8, 0.8, 0.8] for i in range(len(mesh_ls.lines))])
                vis.add_geometry(mesh_ls)
                pcd1 = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector([pos]))
                pcd1.paint_uniform_color([0.0, 1.0, 0.0])
                vis.add_geometry(pcd1)
                joint_idx = joint_type_to_idx[joint_id]
                pcd2 = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(vtx_ori[np.argwhere(attn_per_joint[joint_idx]).squeeze()]))
                pcd2.paint_uniform_color([1.0, 0.0, 0.0])
                vis.add_geometry(pcd2)
                vis.run()
                vis.destroy_window()

        # For debugging purposes.
        # visualize()

        # Save attn per joint.
        np.savetxt(os.path.join(results_folder, f'{model_id}_attn_per_joint.txt'), attn_per_joint, fmt='%d')

        # Save joint type to index mapping.
        with open(os.path.join(results_folder, f'{model_id}_joint_type_to_idx.txt'), 'w') as f:
            for joint_type, idx in joint_type_to_idx.items():
                f.write(f'{joint_type} {idx}\n')

        # Old code for saving attn.
        # np.savetxt(os.path.join(results_folder, f'{model_id}.txt'), attn, fmt='%d')