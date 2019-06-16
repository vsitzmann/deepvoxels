import torch
import torch.nn as nn
import numpy as np
from torch.nn.modules.utils import _ntuple
import util

import torchvision.utils
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from pytorch_prototyping.pytorch_prototyping import *


class FeatureExtractor(nn.Module):
    def __init__(self,
                 nf0,
                 out_channels,
                 input_resolution,
                 output_sidelength):
        super().__init__()

        norm = nn.BatchNorm2d

        num_down_unet = util.num_divisible_by_2(output_sidelength)
        num_downsampling = util.num_divisible_by_2(input_resolution) - num_down_unet

        self.net = nn.Sequential(
            DownsamplingNet([nf0 * (2 ** i) for i in range(num_downsampling)],
                            in_channels=3,
                            use_dropout=False,
                            norm=norm),
            Unet(in_channels=nf0 * (2 ** (num_downsampling-1)),
                 out_channels=out_channels,
                 nf0=nf0 * (2 ** (num_downsampling-1)),
                 use_dropout=False,
                 max_channels=8*nf0,
                 num_down=num_down_unet,
                 norm=norm)
        )

    def forward(self, input):
        return self.net(input)

class RenderingNet(nn.Module):
    def __init__(self,
                 nf0,
                 in_channels,
                 input_resolution,
                 img_sidelength):
        super().__init__()

        num_down_unet = util.num_divisible_by_2(input_resolution)
        num_upsampling = util.num_divisible_by_2(img_sidelength) - num_down_unet

        self.net = [
            Unet(in_channels=in_channels,
                 out_channels=3 if num_upsampling <= 0 else 4*nf0,
                 outermost_linear=True if num_upsampling <= 0 else False,
                 use_dropout=True,
                 dropout_prob=0.1,
                 nf0=nf0*(2**num_upsampling),
                 norm=nn.BatchNorm2d,
                 max_channels=8*nf0,
                 num_down=num_down_unet)
        ]

        if num_upsampling > 0:
            self.net += [
                UpsamplingNet(per_layer_out_ch=num_upsampling * [nf0],
                              in_channels=4 * nf0,
                              upsampling_mode='transpose',
                              use_dropout=True,
                              dropout_prob=0.1),
                Conv2dSame(nf0, out_channels=nf0 // 2, kernel_size=3, bias=False),
                nn.BatchNorm2d(nf0 // 2),
                nn.ReLU(True),
                Conv2dSame(nf0//2, 3, kernel_size=3)
            ]

        self.net += [nn.Tanh()]
        self.net = nn.Sequential(*self.net)

    def forward(self, input):
        return self.net(input)


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

