import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import trimesh
import numpy as np
import glob

from multiprocessing import Pool

import argparse
import traceback

import utils.pcd_utils as pcd_utils

def boundary_sampling_flow(character_names):
    base_dir = '/cluster/falas/xzhang/DT4D_test'
    in_path = os.path.join(base_dir, character_names)
    mesh_path = os.path.join(in_path, 'norm_meshes_seq')

    try:
        if "SPLITS" in in_path or in_path.endswith("json") or in_path.endswith("txt") or in_path.endswith("npz"):
            return

        out_path = os.path.join(in_path, 'flow_seq')
        os.makedirs(out_path,exist_ok=True)
        out_file = os.path.join(out_path, 't-pose_flow_samples.npz')
        test_exit = os.path.join(out_path, '16-pose_flow_samples.npz')
        # print(test_exit)
        if not OVERWRITE and os.path.exists(test_exit):
            print("Skipping", out_path)
            return

        ref_mesh_path = os.path.join(mesh_path, "t-pose_normalized.ply")
        ref_mesh = trimesh.load_mesh(ref_mesh_path, process=False, maintain_order=True)

        # Sample points in tpose, then use the barycentric coords to sample corresponding points in other meshes
        _, faces_ref, bary_coords_ref, _ = pcd_utils.sample_points(
            ref_mesh, sample_num, return_barycentric=True
        )

        num1 = int(sample_num * 0.5)  # On the surface
        num2 = sample_num - num1 # Along the normal
        noise1 = np.zeros((num1, 1))
        noise2 = np.random.rand(num2, 1) - 0.5 # [-0.5, 0.5]
        noise2 = noise2 * args.sigma
        random_noise = np.concatenate([noise1, noise2], axis=0)

        # sampling_info_path = os.path.join(out_path, f"sampling_info_flow_{args.sigma}.npz")
        sampling_info_path = os.path.join(out_path, f"sampling_info_flow.npz")

        np.savez(
                    sampling_info_path,
                    faces_ref=faces_ref,
                    bary_coords_ref=bary_coords_ref,
                    random_noise=random_noise,
                )

        # Load mesh
        mesh = trimesh.load_mesh(os.path.join(mesh_path, 't-pose_normalized.ply'), process=False, maintain_order=True)

        ############################################################################################################
        # Sample points on the mesh using the reference barycentric coordinates computed at the beginning
        ############################################################################################################

        if args.sigma > 0:
            # Along the normal
            points_source, points_triangles = pcd_utils.sample_points_give_bary(
                mesh, faces_ref, bary_coords_ref
            )
            # print(points_source[0])

            normals, is_valid_normal = trimesh.triangles.normals(points_triangles)
            assert np.all(is_valid_normal), "Not all normals were valid"

            points = points_source + random_noise * normals

        else:
            # On the surface
            points, points_triangles = pcd_utils.sample_points_give_bary(
                mesh, faces_ref, bary_coords_ref
            )
            normals, is_valid_normal = trimesh.triangles.normals(points_triangles)
            assert np.all(is_valid_normal), "Not all normals were valid"


        np.savez(out_file, points=points, normals=normals)

        for i in range(1,17):
            mesh = trimesh.load(os.path.join(mesh_path, str(i) + "-pose_normalized.ply"), process=False, maintain_order=True)
            flow_out_file = os.path.join(out_path, str(i) + '-pose_flow_samples.npz')
            if args.sigma > 0:
                # Along the normal
                points_source, points_triangles = pcd_utils.sample_points_give_bary(
                    mesh, faces_ref, bary_coords_ref
                )
                # print(i,points_source[0])

                normals, is_valid_normal = trimesh.triangles.normals(points_triangles)
                assert np.all(is_valid_normal), "Not all normals were valid"

                points = points_source + random_noise * normals

            else:
                # On the surface
                points, points_triangles = pcd_utils.sample_points_give_bary(
                    mesh, faces_ref, bary_coords_ref
                )
                normals, is_valid_normal = trimesh.triangles.normals(points_triangles)
                assert np.all(is_valid_normal), "Not all normals were valid"

            np.savez(flow_out_file, points=points, normals=normals)

        print('Done with {}'.format(out_file))

    except:
        print('\t------------ Error with {}: {}'.format(in_path, traceback.format_exc()))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Run boundary sampling'
    )
    parser.add_argument('-sigma', type=float, required=True)
    parser.add_argument('-t', '-max_threads', dest='max_threads', type=int, default=-1)
    args = parser.parse_args()

    try:
        n_jobs = int(os.environ['SLURM_CPUS_ON_NODE'])
        assert args.max_threads != 0
        if args.max_threads > 0:
            n_jobs = args.max_threads
    except:
        n_jobs = 1

    #####################################################################
    # Set up
    #####################################################################

    OVERWRITE = False
    use_stored_sampling = True
    only_aug = False

    sample_num = 200000

    # ----------------------------------------------------------------- #
    datasets_name = "datasets"
    # ----------------------------------------------------------------- #
    dataset = "deforming4d"
    # ----------------------------------------------------------------- #


    print("DATASET:", dataset)

    split_file_path = './DT4D/train2.lst'
    character_names = sorted([line.rstrip('\n') for line in open(split_file_path)])


    print()
    for c in character_names:
        print(c)
    
    input(f"Continue? {len(character_names)} characters")


    try:
        p = Pool(n_jobs)
        p.map(boundary_sampling_flow, character_names)
    finally:
        p.close()
        p.join()
        