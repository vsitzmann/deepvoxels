import cv2
import numpy as np
import imageio
from skimage import io, transform
from glob import glob
import os
import torch
import util
import pickle as pck
import math
from scipy.linalg import logm, norm
import shutil


def square_crop_img(img):
    min_dim = np.amin(img.shape[:2])
    center_coord = np.array(img.shape[:2]) // 2
    img = img[center_coord[0] - min_dim // 2:center_coord[0] + min_dim // 2,
          center_coord[1] - min_dim // 2:center_coord[1] + min_dim // 2]
    return img


def load_img(filepath, target_size=None, anti_aliasing=True, downsampling_order=None, square_crop=False):
    img = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)

    if img is None:
        print("Error: Path %s invalid" % filepath)
        return None

    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if square_crop:
        img = square_crop_img(img)

    if target_size is not None:
        if downsampling_order == 1:
            img = cv2.resize(img, tuple(target_size), interpolation=cv2.INTER_AREA)
        else:
            img = transform.resize(img, target_size,
                                   order=downsampling_order,
                                   mode='reflect',
                                   clip=False, preserve_range=True,
                                   anti_aliasing=anti_aliasing)
    return img


def load_pose(filename):
    lines = open(filename).read().splitlines()
    if len(lines) == 1:
        pose = np.zeros((4, 4), dtype=np.float32)
        for i in range(16):
            pose[i // 4, i % 4] = lines[0].split(" ")[i]
        return pose.squeeze()
    else:
        lines = [[x[0], x[1], x[2], x[3]] for x in (x.split(" ") for x in lines[:4])]
        return np.asarray(lines).astype(np.float32).squeeze()


def glob_imgs(path):
    imgs = []
    for ext in ['*.png', '*.jpg', '*.JPEG']:
        imgs.extend(glob(os.path.join(path, ext)))
    return imgs


def remove_margin(img, margin):
    return img[margin:-margin, margin:-margin, :]


def read_view_direction_rays(direction_file):
    img = cv2.imread(direction_file, cv2.IMREAD_UNCHANGED).astype(np.float32)
    img -= 40000
    img /= 10000
    return img


def get_pose_img(poses_file):
    pose = load_pose(poses_file)
    img = np.tile(pose.reshape(-1)[None, None, :], (512, 512, 1))
    return img


def process_ray_dirs(pose_dir, target_dir):
    ray_dir = os.path.join(target_dir, 'ray_dirs_high')
    view_dir = os.path.join(target_dir, 'view_dirs_high')

    print(ray_dir)

    for dir in [ray_dir, view_dir]:
        if not os.path.exists(dir):
            os.mkdir(dir)

    all_poses = sorted(glob(os.path.join(pose_dir, '*.txt')))
    full_intrinsic = np.array([[525., 0., 256., 0.], [0., 525., 256., 0], [0., 0, 1, 0], [0, 0, 0, 1]])
    # full_intrinsic = np.array([[525., 0., 319.5, 0.], [0., 525., 239.5, 0], [0., 0, 1, 0], [0, 0, 0, 1]])

    # high_res_intrinsic = np.copy(full_intrinsic)
    # high_res_intrinsic[:2, :3] *= 512. / 480.
    # high_res_intrinsic[:2, 2] = 512. / 480 * 239.5

    # high_res_intrinsic = torch.Tensor(high_res_intrinsic).float()
    high_res_intrinsic = torch.Tensor(full_intrinsic).float()

    ray_min, ray_max, view_min, view_max = 1000., -1., 1000., -1.
    for i, pose_file in enumerate(all_poses):
        print(pose_file)
        pose = load_pose(pose_file)

        view_rays = util.compute_view_directions(high_res_intrinsic,
                                                 torch.from_numpy(pose).squeeze(),
                                                 img_height_width=(512, 512),
                                                 voxel_size=1,
                                                 frustrum_depth=1).squeeze().cpu().permute(1, 2, 0).numpy()
        view_direction = np.tile(pose[:3, 2].squeeze()[None, None, :], (512, 512, 1))
        view_direction /= np.linalg.norm(view_direction, axis=2, keepdims=True)

        # ray_min = min(ray_min, np.amin(view_rays)) # -1.03
        # ray_max = max(ray_max, np.amax(view_rays)) # 1.03
        # view_min = min(view_min, np.amin(view_direction)) # -1.69871
        # view_max = max(view_max, np.amax(view_direction)) # 1.699954

        view_rays *= 10000
        view_rays += 40000

        view_direction *= 10000
        view_direction += 40000

        cv2.imwrite(os.path.join(ray_dir, "%05d.png" % i), view_rays.round().astype(np.uint16))
        cv2.imwrite(os.path.join(view_dir, "%05d.png" % i), view_direction.round().astype(np.uint16))


def get_archimedean_spiral(sphere_radius, origin=np.array([0., 0., 0.])):
    a = 300
    r = sphere_radius
    o = origin

    translations = []

    i = a / 2
    while i > 0.:
        x = r * np.cos(i) * np.cos((-np.pi / 2) + i / a * np.pi)
        y = r * np.sin(i) * np.cos((-np.pi / 2) + i / a * np.pi)
        z = r * - np.sin(-np.pi / 2 + i / a * np.pi)

        xyz = np.array((x,y,z)) + o

        translations.append(xyz)
        i -= a / (2 * 1000.)

    return translations


def interpolate_views(pose_1, pose_2, num_steps=100):
    poses = []
    for i in np.linspace(0., 1., num_steps):
        pose_1_contrib = 1 - i
        pose_2_contrib = i

        # Interpolate the two matrices
        target_pose = pose_1_contrib * pose_1 + pose_2_contrib * pose_2

        # Renormalize the rotation matrix
        target_pose[:3,:3] /= np.linalg.norm(target_pose[:3,:3], axis=0, keepdims=True)
        poses.append(target_pose)

    return poses


def cond_mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def get_nn_ranking(poses):
    # Calculate the ranking of nearest neigbors
    parsed_poses = np.stack([pose[:3,2] for pose in poses], axis=0)
    parsed_poses /= np.linalg.norm(parsed_poses, axis=1, ord=2, keepdims=True)
    cos_sim_mat = parsed_poses.dot(parsed_poses.T)
    np.fill_diagonal(cos_sim_mat, -1.)
    nn_idcs = cos_sim_mat.argsort(axis=1).astype(int)  # increasing order
    cos_sim_mat.sort(axis=1)

    return nn_idcs, cos_sim_mat


def get_filename_no_ext(path):
    return os.path.splitext(os.path.basename(os.path.normpath(path)))[0]


def load_wrld_coords(path):
    coords = load_img(path)
    return coords.astype(np.float32) * 1e-4 - 0.5


def load_depth(path):
    depth = load_img(path)
    if (len(depth.shape) > 2):
        depth = depth[:, :, 0]

    return depth.astype(np.float32) * 1e-4


def interpolate_training_poses(data_root, trgt_root, num_samples=10, num_steps=100):
    np.random.seed(0)

    pose_dir = os.path.join(data_root, 'pose')
    test_pose_dir = os.path.join(trgt_root, 'pose')

    cond_mkdir(test_pose_dir)

    all_pose_paths = sorted(glob(os.path.join(pose_dir, '*.txt')))
    nn_rankings, cos_sim_mat = get_nn_ranking([load_pose(path) for path in all_pose_paths])

    prev_pose_idx = 0

    for sample in range(num_samples):
        if sample:
            rand_idx_1 = prev_pose_idx
        else:
            rand_idx_1 = np.random.choice(len(all_pose_paths))

        pose_1_path = all_pose_paths[rand_idx_1]

        # The second one, sample from the furthest 10 poses
        pose_nns = nn_rankings[rand_idx_1]
        pose_2_idx = pose_nns[np.random.randint(low=1, high=11)] # low is 1 because 0 is the pose itself.
        pose_2_path = all_pose_paths[pose_2_idx]

        prev_pose_idx = pose_2_idx

        print(rand_idx_1, pose_2_idx)

        pose_1_name, pose_2_name = tuple(get_filename_no_ext(path) for path in [pose_1_path, pose_2_path])

        pose_1 = load_pose(pose_1_path)
        pose_2 = load_pose(pose_2_path)

        interpolated_views = interpolate_views(pose_1, pose_2, num_steps)

        for idx, view in enumerate(interpolated_views):
            with open(os.path.join(test_pose_dir, "%06d_%s_%s.txt"%(idx+(sample*num_steps), pose_1_name, pose_2_name)), 'w') as pose_file:
                pose_file.write(' '.join(map(str, view.reshape(-1).tolist())) + '\n')


##################################################
##### Utility function for rotation matrices - from https://github.com/akar43/lsm/blob/b09292c6211b32b8b95043f7daf34785a26bce0a/utils.py #####
##################################################


def quat2rot(q):
    '''q = [w, x, y, z]
    https://en.wikipedia.org/wiki/Rotation_matrix#Quaternion'''
    eps = 1e-5
    w, x, y, z = q
    n = np.linalg.norm(q)
    s = (0 if n < eps else 2.0 / n)
    wx = s * w * x
    wy = s * w * y
    wz = s * w * z
    xx = s * x * x
    xy = s * x * y
    xz = s * x * z
    yy = s * y * y
    yz = s * y * z
    zz = s * z * z
    R = np.array([[1 - (yy + zz), xy - wz,
                   xz + wy], [xy + wz, 1 - (xx + zz), yz - wx],
                  [xz - wy, yz + wx, 1 - (xx + yy)]])
    return R


def rot2quat(M):
    if M.shape[0] < 4 or M.shape[1] < 4:
        newM = np.zeros((4, 4))
        newM[:3, :3] = M[:3, :3]
        newM[3, 3] = 1
        M = newM

    q = np.empty((4, ))
    t = np.trace(M)
    if t > M[3, 3]:
        q[0] = t
        q[3] = M[1, 0] - M[0, 1]
        q[2] = M[0, 2] - M[2, 0]
        q[1] = M[2, 1] - M[1, 2]
    else:
        i, j, k = 0, 1, 2
        if M[1, 1] > M[0, 0]:
            i, j, k = 1, 2, 0
        if M[2, 2] > M[i, i]:
            i, j, k = 2, 0, 1
        t = M[i, i] - (M[j, j] + M[k, k]) + M[3, 3]
        q[i] = t
        q[j] = M[i, j] + M[j, i]
        q[k] = M[k, i] + M[i, k]
        q[3] = M[k, j] - M[j, k]
        q = q[[3, 0, 1, 2]]
    q *= 0.5 / math.sqrt(t * M[3, 3])
    return q


def euler_to_rot(theta):
    R_x = np.array([[1, 0, 0], [0, math.cos(theta[0]), -math.sin(theta[0])],
                    [0, math.sin(theta[0]),
                     math.cos(theta[0])]])

    R_y = np.array([[math.cos(theta[1]), 0,
                     math.sin(theta[1])], [0, 1, 0],
                    [-math.sin(theta[1]), 0,
                     math.cos(theta[1])]])

    R_z = np.array([[math.cos(theta[2]), -math.sin(theta[2]), 0],
                    [math.sin(theta[2]),
                     math.cos(theta[2]), 0], [0, 0, 1]])

    R = np.dot(R_z, np.dot(R_y, R_x))

    return R


def az_el_to_rot(az, el):
    corr_mat = np.array([[0, 0, -1], [1, 0, 0], [0, -1, 0]])
    inv_corr_mat = np.linalg.inv(corr_mat)

    def R_x(theta):
        return np.array([[1, 0, 0], [0, math.cos(theta),
                                     math.sin(theta)],
                         [0, -math.sin(theta),
                          math.cos(theta)]])

    def R_y(theta):
        return np.array([[math.cos(theta), 0, -math.sin(theta)], [0, 1, 0],
                         [math.sin(theta), 0,
                          math.cos(theta)]])

    def R_z(theta):
        return np.array([[math.cos(theta), -math.sin(theta), 0],
                         [math.sin(theta), math.cos(theta), 0], [0, 0, 1]])

    Rmat = np.matmul(R_x(-el * math.pi / 180), R_y(-az * math.pi / 180))
    return np.matmul(Rmat, inv_corr_mat)


def rand_euler_rotation_matrix(nmax=10):
    euler = (np.random.uniform(size=(3, )) - 0.5) * nmax * 2 * math.pi / 360.0
    Rmat = euler_to_rot(euler)
    return Rmat, euler * 180 / math.pi


def rot_mag(R):
    angle = (1.0 / math.sqrt(2)) * \
        norm(logm(R), 'fro') * 180 / (math.pi)
    return angle


def add_noise(pose, nmax=10):
    rand_rot, euler = rand_euler_rotation_matrix(float(nmax))
    pose[:3,:3] = rand_rot.dot(pose[:3,:3])
    return pose


def create_noisy_poses(data_root, trgt_root, ang_noise_magnitude):
    np.random.seed(0)

    pose_dir = os.path.join(data_root, 'pose')
    trgt_pose_dir = os.path.join(trgt_root, 'pose')

    cond_mkdir(trgt_pose_dir)

    all_pose_paths = sorted(glob(os.path.join(pose_dir, '*.txt')))
    all_poses = [(os.path.basename(path), load_pose(path)) for path in all_pose_paths]

    for fname, pose in all_poses:
        noisy_pose = add_noise(pose, ang_noise_magnitude)
        with open(os.path.join(trgt_pose_dir, fname), 'w') as pose_file:
            matrix_flat = []
            for j in range(4):
                for k in range(4):
                    matrix_flat.append(noisy_pose[j][k])
            pose_file.write(' '.join(map(str, matrix_flat)) + '\n')


def invert_poses(data_root, trgt_root):
    pose_dir = os.path.join(data_root, 'pose')
    trgt_pose_dir = os.path.join(trgt_root, 'pose')

    cond_mkdir(trgt_pose_dir)

    all_pose_paths = sorted(glob(os.path.join(pose_dir, '*.txt')))
    all_poses = [(os.path.basename(path), load_pose(path)) for path in all_pose_paths]

    for fname, pose in all_poses:
        inverted_pose = np.copy(pose)
        inverted_pose[:3,3] *= -1
        inverted_pose[:3,2] *= -1
        with open(os.path.join(trgt_pose_dir, fname), 'w') as pose_file:
            matrix_flat = []
            for j in range(4):
                for k in range(4):
                    matrix_flat.append(inverted_pose[j][k])
            pose_file.write(' '.join(map(str, matrix_flat)) + '\n')


def nearest_neighbor_baseline(train_dir, test_dir, target_dir):
    train_pose_dir, test_pose_dir = [os.path.join(dir, 'pose') for dir in [train_dir, test_dir]]
    train_img_dir = os.path.join(train_dir, 'rgb')

    nns = util.get_nearest_neighbors_pose(train_pose_dir, test_pose_dir, metric='cos', sampling_pattern='all')

    for i in range(len(nns)):
        print(os.path.join(train_img_dir, '%06d.png'%nns[i]))
        img = load_img(os.path.join(train_img_dir, '%06d.png'%nns[i]),
                       target_size=(512,512),
                       square_crop=True,
                       downsampling_order=1)[:,:,::-1]
        cv2.imwrite(os.path.join(target_dir, '%06d.png'%i), img)

if __name__ == '__main__':
    # Adds noise to training poses.
    # data_root = '/home/vincent/data/deepvoxels/pedestal'
    # trgt_root = data_root + '_5_deg_noise'
    # cond_mkdir(trgt_root)
    # create_noisy_poses(data_root, trgt_root, 5)

    for object in ['coffee', 'globe', 'stanford_fountain', 'dyck']:
        # Interpolates training poses to generate test trajectories
        base_dir ='/home/vincent/data/deep_voxels/data/real_captures'
        train_root = os.path.join(base_dir, object)
        trgt_root = train_root + '_test'
        cond_mkdir(trgt_root)
        interpolate_training_poses(train_root, trgt_root)

        # Copy the intrinsics file
        shutil.copy(os.path.join(train_root,'intrinsics.txt'),
                    os.path.join(trgt_root, 'intrinsics.txt'))

        # Compute nearest neighbor baseline
        nn_root = os.path.join(base_dir, object + '_nn')
        cond_mkdir(nn_root)
        nearest_neighbor_baseline(train_root, trgt_root, nn_root)
