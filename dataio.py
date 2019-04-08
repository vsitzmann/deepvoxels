import os
import torch
import numpy as np
from torch.utils.data import Dataset
from glob import glob
from copy import deepcopy
import data_util
import matplotlib.pyplot as plt


class NovelViewTriplets():
    def __init__(self,
                 root_dir,
                 img_size,
                 sampling_pattern):
        super().__init__()

        self.img_size = img_size

        self.color_dir = os.path.join(root_dir, 'rgb')
        self.pose_dir = os.path.join(root_dir, 'pose')

        if not os.path.isdir(self.color_dir):
            print("Error! root dir is wrong")
            return

        self.all_color = sorted(data_util.glob_imgs(self.color_dir))
        self.all_poses = sorted(glob(os.path.join(self.pose_dir, '*.txt')))

        # Subsample the trajectory for training / test set split as well as the result matrix
        file_lists = [self.all_color, self.all_poses]

        if sampling_pattern != 'all':
            if sampling_pattern.split('_')[0] == 'skip':
                skip_val = int(sampling_pattern.split('_')[-1])

                for i in range(len(file_lists)):
                    dummy_list = deepcopy(file_lists[i])
                    file_lists[i].clear()
                    file_lists[i].extend(dummy_list[::skip_val + 1])
            else:
                print("Unknown sampling pattern!")
                return None

        # Buffer files
        print("Buffering files...")
        self.all_views = []
        for i in range(self.__len__()):
            if not i % 10:
                print(i)
            self.all_views.append(self.read_view_tuple(i))

        # Calculate the ranking of nearest neigbors
        self.nn_idcs, _ = data_util.get_nn_ranking([data_util.load_pose(pose) for pose in self.all_poses])

        print("*" * 100)
        print("Sampling pattern ", sampling_pattern)
        print("Image size ", self.img_size)
        print("*" * 100)

    def load_rgb(self, path):
        img = data_util.load_img(path, square_crop=True, downsampling_order=1, target_size=self.img_size)
        img = img[:, :, :3].astype(np.float32) / 255. - 0.5
        img = img.transpose(2,0,1)
        return img

    def read_view_tuple(self, idx):
        gt_rgb = self.load_rgb(self.all_color[idx])
        pose = data_util.load_pose(self.all_poses[idx])

        this_view = {'gt_rgb': torch.from_numpy(gt_rgb),
                     'pose': torch.from_numpy(pose)}
        return this_view

    def __len__(self):
        return len(self.all_color)

    def __getitem__(self, idx):
        trgt_views = []

        # Read one target pose and its nearest neighbor
        trgt_views.append(self.all_views[idx])
        nearest_view = self.all_views[self.nn_idcs[idx][-np.random.randint(low=1, high=5)]]

        # The second target pose is a random one
        trgt_views.append(self.all_views[np.random.choice(len(self))])

        return trgt_views, nearest_view


class TestDataset():
    def __init__(self,
                 pose_dir):
        super().__init__()

        all_pose_paths = sorted(glob(os.path.join(pose_dir, '*.txt')))
        self.all_poses = [torch.from_numpy(data_util.load_pose(path)) for path in all_pose_paths]

    def __len__(self):
        return len(self.all_poses)

    def __getitem__(self, idx):
        return self.all_poses[idx]
