import os, struct, math
import numpy as np
import torch
from glob import glob
import data_util

import shlex
import subprocess

import torch.nn.functional as F

def backproject(ux, uy, depth, intrinsic):
    '''Given a point in pixel coordinates plus depth gives the coordinates of the imaged point in camera coordinates
    '''
    x = (ux - intrinsic[0][2]) / intrinsic[0][0]
    y = (uy - intrinsic[1][2]) / intrinsic[1][1]
    return torch.stack([depth * x, depth * y, depth, torch.ones_like(depth)], dim=0)


def parse_intrinsics(filepath, trgt_sidelength, invert_y=False):
    # Get camera intrinsics
    with open(filepath, 'r') as file:
        f, cx, cy = list(map(float, file.readline().split()))[:3]
        grid_barycenter = torch.Tensor(list(map(float, file.readline().split())))
        near_plane = float(file.readline())
        scale = float(file.readline())
        height, width = map(float, file.readline().split())

        try:
            world2cam_poses = int(file.readline())
        except ValueError:
            world2cam_poses = None

    if world2cam_poses is None:
        world2cam_poses = False

    world2cam_poses = bool(world2cam_poses)

    cx = cx / width * trgt_sidelength
    cy = cy / height * trgt_sidelength
    f = trgt_sidelength / height * f

    fx = f
    if invert_y:
        fy = -f
    else:
        fy = f

    # Build the intrinsic matrices
    full_intrinsic = np.array([[fx, 0., cx, 0.],
                               [0., fy, cy, 0],
                               [0., 0, 1, 0],
                               [0, 0, 0, 1]])

    return full_intrinsic, grid_barycenter, scale, near_plane, world2cam_poses


def resize2d(img, size):
    return F.adaptive_avg_pool2d(img, size[2:])


def compute_warp_idcs(cam_1_intrinsic,
                      cam_2_intrinsic,
                      img_1_pose,
                      img_1_depth,
                      img_2_pose,
                      img_2_depth):
    cam_1_intrinsic = cam_1_intrinsic.squeeze().cuda()
    cam_2_intrinsic = cam_2_intrinsic.squeeze().cuda()

    img_1_pose = img_1_pose.squeeze().cuda()
    img_2_pose = img_2_pose.squeeze().cuda()

    img_1_depth = img_1_depth.squeeze().cuda()
    img_2_depth = img_2_depth.squeeze().cuda()

    # Get the new size
    side_length = img_1_depth.shape[0]

    # Get camera coordinates of pixels in camera 1
    pixel_range = torch.arange(0, side_length)
    xx, yy = torch.meshgrid([pixel_range, pixel_range])

    xx = xx.contiguous().view(-1).float().cuda()
    yy = yy.contiguous().view(-1).float().cuda()

    img_1_cam_coords = backproject(yy,
                                   xx,
                                   img_1_depth.contiguous().view(-1),
                                   cam_1_intrinsic)

    # Convert to world coordinates
    world_coords = torch.mm(img_1_pose, img_1_cam_coords)

    # Convert to cam 2 coordinates
    trgt_coords = torch.mm(torch.inverse(img_2_pose), world_coords)
    trgt_coords = torch.mm(cam_2_intrinsic, trgt_coords)

    # Get the depths in the target camera frame
    transformed_depths = trgt_coords[2, :].clone()

    # z-divide.
    trgt_coords /= trgt_coords[2:3, :] + 1e-9
    trgt_idcs = torch.round(trgt_coords[:2]).long()

    # Mask out everything outside the image boundaries
    mask_img_bounds = (torch.ge(trgt_idcs[0], 0) *
                       torch.ge(trgt_idcs[1], 0))
    mask_img_bounds = (mask_img_bounds *
                       torch.lt(trgt_idcs[0], side_length) *
                       torch.lt(trgt_idcs[1], side_length))

    if not mask_img_bounds.any():
        print('Nothing in warped image')
        return None

    valid_trgt_idcs = trgt_idcs[:, mask_img_bounds]

    gt_depths = img_2_depth[valid_trgt_idcs[1, :], valid_trgt_idcs[0, :]]
    not_occluded = (torch.abs(gt_depths.detach() - transformed_depths[mask_img_bounds].detach()) < 0.05)
    # not_occluded = gt_depths < 1000.

    if not not_occluded.any():
        print('Nothing unoccluded')
        return None

    # Get the final coordinates
    valid_xx = xx[mask_img_bounds][not_occluded].long()
    valid_yy = yy[mask_img_bounds][not_occluded].long()
    valid_trgt_coords = trgt_coords[:, mask_img_bounds][:, not_occluded]

    return torch.stack([valid_xx, valid_yy], dim=0), valid_trgt_coords


def concat_pose(feature_map, pose):
    feat_map = torch.cat([feature_map, pose.squeeze()[None, :, None, None].repeat(1, 1, 64, 64)], dim=1)
    return feat_map


def num_divisible_by_2(number):
    i = 0
    while not number % 2:
        number = number // 2
        i += 1

    return i


def compute_view_directions(intrinsic,
                            cam2world,
                            img_height_width,
                            voxel_size,
                            frustrum_depth=1,
                            near_plane=np.sqrt(3) / 2):
    xx, yy, zz = torch.meshgrid([torch.arange(0, img_height_width[1]),
                                 torch.arange(0, img_height_width[0]),
                                 torch.arange(0, frustrum_depth)])

    coords = torch.stack([xx, yy, zz, torch.zeros_like(xx)], dim=0).float()

    coords[2] *= voxel_size
    coords[2] += near_plane

    coords[0] = (coords[0] - intrinsic[0][2]) / intrinsic[0][0]
    coords[1] = (coords[1] - intrinsic[1][2]) / intrinsic[1][1]
    coords[:2] *= coords[2]

    coords = coords.view(4, -1)
    world_coords = torch.mm(cam2world, coords)[:3]
    world_coords /= world_coords.norm(2, dim=0, keepdim=True)
    world_coords = world_coords.view(3, img_height_width[1], img_height_width[0], frustrum_depth)
    return world_coords


# util for saving tensors, for debug purposes
def write_array_to_file(tensor, filename):
    sz = tensor.shape
    with open(filename, 'wb') as f:
        f.write(struct.pack('Q', sz[0]))
        f.write(struct.pack('Q', sz[1]))
        f.write(struct.pack('Q', sz[2]))
        tensor.tofile(f)


def read_lines_from_file(filename):
    assert os.path.isfile(filename)
    lines = open(filename).read().splitlines()
    return lines


# create camera intrinsics
def make_intrinsic(fx, fy, mx, my):
    intrinsic = torch.eye(4)
    intrinsic[0][0] = fx
    intrinsic[1][1] = fy
    intrinsic[0][2] = mx
    intrinsic[1][2] = my
    return intrinsic


# create camera intrinsics
def adjust_intrinsic(intrinsic, intrinsic_image_dim, image_dim):
    if intrinsic_image_dim == image_dim:
        return intrinsic
    resize_width = int(math.floor(image_dim[1] * float(intrinsic_image_dim[0]) / float(intrinsic_image_dim[1])))
    intrinsic[0, 0] *= float(resize_width) / float(intrinsic_image_dim[0])
    intrinsic[1, 1] *= float(image_dim[1]) / float(intrinsic_image_dim[1])
    # account for cropping here
    intrinsic[0, 2] *= float(image_dim[0] - 1) / float(intrinsic_image_dim[0] - 1)
    intrinsic[1, 2] *= float(image_dim[1] - 1) / float(intrinsic_image_dim[1] - 1)
    return intrinsic


def get_sample_files(samples_path):
    files = [f for f in os.listdir(samples_path) if f.endswith('.sample')]  # and os.path.isfile(join(samples_path, f))]
    return files


def get_sample_files_for_scene(scene, samples_path):
    files = [f for f in os.listdir(samples_path) if
             f.startswith(scene) and f.endswith('.sample')]  # and os.path.isfile(join(samples_path, f))]
    print('found ', len(files), ' for ', os.path.join(samples_path, scene))
    return files


def cond_mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def load_pose(filename):
    assert os.path.isfile(filename)
    pose = torch.Tensor(4, 4)
    lines = open(filename).read().splitlines()
    assert len(lines) == 4
    lines = [[x[0], x[1], x[2], x[3]] for x in (x.split(" ") for x in lines)]
    return torch.from_numpy(np.asarray(lines).astype(np.float32))


def expand_to_feature_map(torch_tensor, img_size):
    return torch_tensor[:, :, None, None].repeat(1, 1, img_size[0], img_size[1])


def normalize(img):
    return (img - img.min()) / (img.max() - img.min())


def write_image(writer, name, img, iter):
    writer.add_image(name, normalize(img.permute([0, 3, 1, 2])), iter)


def print_network(net):
    model_parameters = filter(lambda p: p.requires_grad, net.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print("%d" % params)


def custom_load(model, path, discriminator=None):
    whole_dict = torch.load(path)
    model.load_state_dict(whole_dict['model'])

    if discriminator:
        discriminator.load_state_dict(whole_dict['discriminator'])


def custom_save(model, path, discriminator=None):
    whole_dict = {'model': model.state_dict()}
    if discriminator:
        whole_dict.update({'discriminator': discriminator.state_dict()})

    torch.save(whole_dict, path)


def get_nearest_neighbors_pose(train_pose_dir, test_pose_dir, sampling_pattern='skip_2', metric='cos'):
    if sampling_pattern != 'all':
        skip_val = int(sampling_pattern.split('_')[-1])
    else:
        skip_val = 0

    train_pose_files = sorted(glob(os.path.join(train_pose_dir, '*.txt')))
    idcs = list(range(len(train_pose_files)))[::skip_val + 1]
    train_pose_files = train_pose_files[::skip_val + 1]

    test_pose_files = sorted(glob(os.path.join(test_pose_dir, '*.txt')))

    train_poses = np.stack([data_util.load_pose(pose)[:3, 3] for pose in train_pose_files], axis=0)
    train_poses /= np.linalg.norm(train_poses, axis=1, keepdims=True)

    test_poses = np.stack([data_util.load_pose(pose)[:3, 3] for pose in test_pose_files], axis=0)
    test_poses /= np.linalg.norm(test_poses, axis=1, keepdims=True)

    if metric == 'cos':
        cos_distance_mat = test_poses.dot(train_poses.T)  # nxn matrix of cosine distances
        nn_idcs = [idcs[int(val)] for val in np.argmax(cos_distance_mat, axis=1)]
    elif metric == 'l2':
        l2_distance_mat = np.linalg.norm(test_poses[:, None, :] - train_poses[None, :, :], axis=2)
        nn_idcs = [idcs[int(val)] for val in np.argmin(l2_distance_mat, axis=1)]

    return nn_idcs
