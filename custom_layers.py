import torch
import torch.nn as nn
import numpy as np
from torch.nn.modules.utils import _ntuple
import util

import torchvision.utils
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from pytorch_prototyping.pytorch_prototyping import *


class IntegrationNet(torch.nn.Module):
    '''The 3D integration net integrating new observations into the Deepvoxels grid.
    '''
    def __init__(self, nf0, coord_conv, use_dropout, per_feature, grid_dim):
        super().__init__()

        self.coord_conv = coord_conv
        if self.coord_conv:
            in_channels = nf0 + 3
        else:
            in_channels = nf0

        if per_feature:
            weights_channels = nf0
        else:
            weights_channels = 1

        self.use_dropout = use_dropout

        self.new_integration = nn.Sequential(
            nn.ReplicationPad3d(1),
            nn.Conv3d(in_channels, nf0, kernel_size=3, padding=0, bias=True),
            nn.Dropout2d(0.2)
        )

        self.old_integration = nn.Sequential(
            nn.ReplicationPad3d(1),
            nn.Conv3d(in_channels, nf0, kernel_size=3, padding=0, bias=False),
            nn.Dropout2d(0.2)
        )

        self.update_old_net = nn.Sequential(
            nn.ReplicationPad3d(1),
            nn.Conv3d(in_channels, weights_channels, kernel_size=3, padding=0, bias=True),
        )
        self.update_new_net = nn.Sequential(
            nn.ReplicationPad3d(1),
            nn.Conv3d(in_channels, weights_channels, kernel_size=3, padding=0, bias=False),
        )

        self.reset_old_net = nn.Sequential(
            nn.ReplicationPad3d(1),
            nn.Conv3d(in_channels, weights_channels, kernel_size=3, padding=0, bias=True),
        )
        self.reset_new_net = nn.Sequential(
            nn.ReplicationPad3d(1),
            nn.Conv3d(in_channels, weights_channels, kernel_size=3, padding=0, bias=False),
        )

        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

        coord_conv_volume = np.mgrid[-grid_dim // 2:grid_dim // 2,
                                     -grid_dim // 2:grid_dim // 2,
                                     -grid_dim // 2:grid_dim // 2]

        coord_conv_volume = np.stack(coord_conv_volume, axis=0).astype(np.float32)
        coord_conv_volume = coord_conv_volume / grid_dim
        self.coord_conv_volume = torch.Tensor(coord_conv_volume).float().cuda()[None, :, :, :, :]
        self.counter = 0

    def forward(self, new_observation, old_state, writer):

        old_state_coord = torch.cat([old_state, self.coord_conv_volume], dim=1)
        new_observation_coord = torch.cat([new_observation, self.coord_conv_volume], dim=1)

        reset = self.sigmoid(self.reset_old_net(old_state_coord) + self.reset_new_net(new_observation_coord))
        update = self.sigmoid(self.update_old_net(old_state_coord) + self.update_new_net(new_observation_coord))

        final = self.relu(self.new_integration(new_observation_coord) + self.old_integration(reset * old_state_coord))

        if not self.counter % 100:
            # Plot the volumes
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            update_values = update.mean(dim=1).squeeze().cpu().detach().numpy()
            x, y, z = np.where(update_values)
            x, y, z = x[::3], y[::3], z[::3]
            ax.scatter(x, y, z, s=update_values[x, y, z] * 5)

            writer.add_figure("update_gate",
                              fig,
                              self.counter,
                              close=True)

            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            reset_values = reset.mean(dim=1).squeeze().cpu().detach().numpy()
            x, y, z = np.where(reset_values)
            x, y, z = x[::3], y[::3], z[::3]
            ax.scatter(x, y, z, s=reset_values[x, y, z] * 5)
            writer.add_figure("reset_gate",
                              fig,
                              self.counter,
                              close=True)
        self.counter += 1

        result = ((1 - update) * old_state + update * final)
        return result


class OcclusionNet(nn.Module):
    '''The Occlusion Module predicts visibility scores for each voxel across a ray, allowing occlusion reasoning
    via a convex combination of voxels along each ray.
    '''
    def __init__(self, nf0, occnet_nf, frustrum_dims):
        super().__init__()

        self.occnet_nf = occnet_nf

        self.frustrum_depth = frustrum_dims[-1]
        depth_coords = torch.arange(-self.frustrum_depth // 2,
                                    self.frustrum_depth // 2)[None, None, :, None, None].float().cuda() / self.frustrum_depth
        self.depth_coords = depth_coords.repeat(1, 1, 1, frustrum_dims[0], frustrum_dims[0])

        self.occlusion_prep = nn.Sequential(
            Conv3dSame(nf0+1, self.occnet_nf, kernel_size=3, bias=False),
            nn.BatchNorm3d(self.occnet_nf),
            nn.ReLU(True),
        )

        num_down = min(util.num_divisible_by_2(self.frustrum_depth),
                       util.num_divisible_by_2(frustrum_dims[0]))

        self.occlusion_net = Unet3d(in_channels=self.occnet_nf,
                                    out_channels=self.occnet_nf,
                                    nf0=self.occnet_nf,
                                    num_down=num_down,
                                    max_channels=4*self.occnet_nf,
                                    outermost_linear=False)

        self.softmax_net = nn.Sequential(
            Conv3dSame(2*self.occnet_nf +1, 1, kernel_size=3, bias=True),
            nn.Softmax(dim=2),
        )

    def forward(self,
                novel_img_frustrum):
        frustrum_feats_depth = torch.cat((self.depth_coords, novel_img_frustrum), dim=1)

        occlusion_prep = self.occlusion_prep(frustrum_feats_depth)
        frustrum_feats = self.occlusion_net(occlusion_prep)
        frustrum_weights = self.softmax_net(torch.cat((occlusion_prep, frustrum_feats, self.depth_coords), dim=1))

        depth_map = (self.depth_coords * frustrum_weights).sum(dim=2)

        return frustrum_weights, depth_map

