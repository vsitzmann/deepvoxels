"""Part of this script is adapted from the Colmap codebase (scripts/python), which uses the new BSD license. Disclaimer:

Copyright (c) 2018, ETH Zurich and UNC Chapel Hill.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.

    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.

    * Neither the name of ETH Zurich and UNC Chapel Hill nor the names of
      its contributors may be used to endorse or promote products derived
      from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.

Author: Johannes L. Schoenberger (jsch-at-demuc-dot-de)
"""
import argparse

import shlex
import subprocess
import struct
import collections
import os
import numpy as np

# params
parser = argparse.ArgumentParser()

# data paths
parser.add_argument('--img_dir', type=str, required=True, help='path to file list of h5 train data')
parser.add_argument('--trgt_dir', required=True, help='path to file list of h5 train data')
parser.add_argument('--dense', action='store_true', default=False, help='#images')

def cond_mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def bundle_adjust(frame_dir, target_dir, dense):
    colmap_template = 'colmap automatic_reconstructor --dense {} --quality low --single_camera 1 --workspace_path {} --image_path {}'

    colmap_cmd = colmap_template.format(1 if dense else 0, target_dir, frame_dir)
    args = shlex.split(colmap_cmd)
    p = subprocess.Popen(args)
    p.wait()


def write_poses(images, target_dir):
    for _, image in images.items():
        name = os.path.splitext(os.path.basename(image.name))[0]
        rot = image.qvec2rotmat()
        trans = image.tvec.reshape(3,1)

        full_pose = np.concatenate((rot, trans), axis=1)
        full_pose = np.concatenate((full_pose, np.array([[0, 0, 0, 1]])), axis=0)
        full_pose = np.linalg.inv(full_pose)

        with open(os.path.join(target_dir, name + '.txt'), 'w') as file:
            file.write(' '.join(map(str, full_pose.reshape(-1).tolist())) + '\n')


def write_intrinsic(cameras, target_dir):
    all_params = []
    for _, camera in cameras.items():
        all_params.append(camera.params)

    example_cam = cameras[list(cameras.keys())[0]]
    height, width = example_cam.height, example_cam.width

    params = np.mean(all_params, axis=0)
    with open(os.path.join(target_dir, 'intrinsics.txt'), 'w') as file:
        file.write(' '.join(map(str, params.reshape(-1).tolist())) + '\n')
        file.write(' '.join(map(str, [0.,0.,0.])) + '\n')
        file.write(str(1.) + '\n')
        file.write(str(height) + ' ' + str(width) + '\n')
        file.write(str(cameras[list(cameras.keys())[0]].model))

""" From Colmap python scripts
"""
BaseImage = collections.namedtuple(
    "Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"])

CameraModel = collections.namedtuple(
    "CameraModel", ["model_id", "model_name", "num_params"])

Camera = collections.namedtuple(
    "Camera", ["id", "model", "width", "height", "params"])



CAMERA_MODELS = {
    CameraModel(model_id=0, model_name="SIMPLE_PINHOLE", num_params=3),
    CameraModel(model_id=1, model_name="PINHOLE", num_params=4),
    CameraModel(model_id=2, model_name="SIMPLE_RADIAL", num_params=4),
    CameraModel(model_id=3, model_name="RADIAL", num_params=5),
    CameraModel(model_id=4, model_name="OPENCV", num_params=8),
    CameraModel(model_id=5, model_name="OPENCV_FISHEYE", num_params=8),
    CameraModel(model_id=6, model_name="FULL_OPENCV", num_params=12),
    CameraModel(model_id=7, model_name="FOV", num_params=5),
    CameraModel(model_id=8, model_name="SIMPLE_RADIAL_FISHEYE", num_params=4),
    CameraModel(model_id=9, model_name="RADIAL_FISHEYE", num_params=5),
    CameraModel(model_id=10, model_name="THIN_PRISM_FISHEYE", num_params=12)
}
CAMERA_MODEL_IDS = dict([(camera_model.model_id, camera_model) for camera_model in CAMERA_MODELS])


def qvec2rotmat(qvec):
    return np.array([
        [1 - 2 * qvec[2] ** 2 - 2 * qvec[3] ** 2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1] ** 2 - 2 * qvec[3] ** 2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1] ** 2 - 2 * qvec[2] ** 2]])


class Image(BaseImage):
    def qvec2rotmat(self):
        return qvec2rotmat(self.qvec)


def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
    """Read and unpack the next bytes from a binary file.
    :param fid:
    :param num_bytes: Sum of combination of {2, 4, 8}, e.g. 2, 6, 16, 30, etc.
    :param format_char_sequence: List of {c, e, f, d, h, H, i, I, l, L, q, Q}.
    :param endian_character: Any of {@, =, <, >, !}
    :return: Tuple of read and unpacked values.
    """
    data = fid.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, data)


def read_images_binary(path_to_model_file):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadImagesBinary(const std::string& path)
        void Reconstruction::WriteImagesBinary(const std::string& path)
    """
    images = {}
    with open(path_to_model_file, "rb") as fid:
        num_reg_images = read_next_bytes(fid, 8, "Q")[0]
        for image_index in range(num_reg_images):
            binary_image_properties = read_next_bytes(
                fid, num_bytes=64, format_char_sequence="idddddddi")
            image_id = binary_image_properties[0]
            qvec = np.array(binary_image_properties[1:5])
            tvec = np.array(binary_image_properties[5:8])
            camera_id = binary_image_properties[8]
            image_name = ""
            current_char = read_next_bytes(fid, 1, "c")[0]
            while current_char != b"\x00":  # look for the ASCII 0 entry
                image_name += current_char.decode("utf-8")
                current_char = read_next_bytes(fid, 1, "c")[0]
            num_points2D = read_next_bytes(fid, num_bytes=8,
                                           format_char_sequence="Q")[0]
            x_y_id_s = read_next_bytes(fid, num_bytes=24 * num_points2D,
                                       format_char_sequence="ddq" * num_points2D)
            xys = np.column_stack([tuple(map(float, x_y_id_s[0::3])),
                                   tuple(map(float, x_y_id_s[1::3]))])
            point3D_ids = np.array(tuple(map(int, x_y_id_s[2::3])))
            images[image_id] = Image(
                id=image_id, qvec=qvec, tvec=tvec,
                camera_id=camera_id, name=image_name,
                xys=xys, point3D_ids=point3D_ids)
    return images

def read_cameras_binary(path_to_model_file):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::WriteCamerasBinary(const std::string& path)
        void Reconstruction::ReadCamerasBinary(const std::string& path)
    """
    cameras = {}
    with open(path_to_model_file, "rb") as fid:
        num_cameras = read_next_bytes(fid, 8, "Q")[0]
        for camera_line_index in range(num_cameras):
            camera_properties = read_next_bytes(
                fid, num_bytes=24, format_char_sequence="iiQQ")
            camera_id = camera_properties[0]
            model_id = camera_properties[1]
            model_name = CAMERA_MODEL_IDS[camera_properties[1]].model_name
            width = camera_properties[2]
            height = camera_properties[3]
            num_params = CAMERA_MODEL_IDS[model_id].num_params
            params = read_next_bytes(fid, num_bytes=8*num_params,
                                     format_char_sequence="d"*num_params)
            cameras[camera_id] = Camera(id=camera_id,
                                        model=model_name,
                                        width=width,
                                        height=height,
                                        params=np.array(params))
        assert len(cameras) == num_cameras
    return cameras


def read_poses(colmap_workspace):
    '''Parses the colmap "images.bin" file

    :param colmap_workspace:
    :return:
    '''
    images = read_images_binary(os.path.join(colmap_workspace, 'sparse', '0', "images.bin"))
    return images

def read_cameras(colmap_workspace):
    cameras = read_cameras_binary(os.path.join(colmap_workspace, 'sparse', '0', "cameras.bin"))
    return cameras



if __name__ == '__main__':
    opt = parser.parse_args()
    print('\n'.join(["%s: %s" % (key, value) for key, value in vars(opt).items()]))
    cond_mkdir(opt.trgt_dir)

    reconst_dir = os.path.join(opt.trgt_dir, 'reconstruction')
    pose_dir = os.path.join(opt.trgt_dir, 'pose')

    cond_mkdir(reconst_dir)
    cond_mkdir(pose_dir)

    print("Bundle Adjusting")
    bundle_adjust(opt.img_dir, reconst_dir, opt.dense)
    print("Extracting poses")
    images = read_poses(reconst_dir)
    print("Writing Poses")
    write_poses(images, pose_dir)
    print("Extracting intrinsics")
    cameras = read_cameras(reconst_dir)
    print("Writing intrinsics")
    write_intrinsic(cameras, opt.trgt_dir)
