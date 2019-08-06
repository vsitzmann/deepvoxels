import argparse
import os, time, datetime

import torch
from torch import nn
import torchvision
import numpy as np
import cv2

from dataio import *
from torch.utils.data import DataLoader

from deep_voxels import DeepVoxels
from projection import ProjectionHelper

from tensorboardX import SummaryWriter
from losses import *
from data_util import *
import util

import time

parser = argparse.ArgumentParser()

parser.add_argument('--train_test', type=str, required=True,
                    help='Whether to run training or testing. Options are \"train\" or \"test\".')
parser.add_argument('--data_root', required=True,
                    help='Path to directory that holds the object data. See dataio.py for directory structure etc..')
parser.add_argument('--logging_root', required=True,
                    help='Path to directory where to write tensorboard logs and checkpoints.')

parser.add_argument('--experiment_name', type=str, default='', help='(optional) Name for experiment.')
parser.add_argument('--max_epoch', type=int, default=400, help='Maximum number of epochs to train for.')
parser.add_argument('--lr', type=float, default=0.0004, help='Learning rate.')
parser.add_argument('--l1_weight', type=float, default=200, help='Weight of l1 loss.')
parser.add_argument('--sampling_pattern', type=str, default='all', required=False,
                    help='Whether to use \"all\" images or whether to skip n images (\"skip_1\" picks every 2nd image.')

parser.add_argument('--img_sidelength', type=int, default=512,
                    help='Sidelength of generated images. Default 512. Only less than native resolution of images is recommended.')

parser.add_argument('--no_occlusion_net', action='store_true', default=False,
                    help='Disables occlusion net and replaces it with a fully convolutional 2d net.')
parser.add_argument('--num_trgt', type=int, default=2, required=False,
                    help='How many novel views will be generated at training time.')

parser.add_argument('--checkpoint', default='',
                    help='Path to a checkpoint to load model weights from.')
parser.add_argument('--start_epoch', type=int, default=0,
                    help='Start epoch')

parser.add_argument('--grid_dim', type=int, default=32,
                    help='Grid sidelength. Default 32.')
parser.add_argument('--num_grid_feats', type=int, default=64,
                    help='Number of features stored in each voxel.')
parser.add_argument('--nf0', type=int, default=64,
                    help='Number of features in outermost layer of U-Net architectures.')
parser.add_argument('--near_plane', type=float, default=np.sqrt(3)/2,
                    help='Position of the near plane.')

opt = parser.parse_args()
print('\n'.join(["%s: %s" % (key, value) for key, value in vars(opt).items()]))

device = torch.device('cuda')

input_image_dims = [opt.img_sidelength, opt.img_sidelength]
proj_image_dims = [64, 64] # Height, width of 2d feature map used for lifting and rendering.

# Read origin of grid, scale of each voxel, and near plane
_, grid_barycenter, scale, near_plane, _ = \
    util.parse_intrinsics(os.path.join(opt.data_root, 'intrinsics.txt'), trgt_sidelength=input_image_dims[0])

if near_plane == 0.0:
    near_plane = opt.near_plane

# Read intrinsic matrix for lifting and projection
lift_intrinsic = util.parse_intrinsics(os.path.join(opt.data_root, 'intrinsics.txt'),
                                       trgt_sidelength=proj_image_dims[0])[0]
proj_intrinsic = lift_intrinsic

# Set up scale and world coordinates of voxel grid
voxel_size = (1. / opt.grid_dim) * scale
grid_origin = torch.tensor(np.eye(4)).float().to(device).squeeze()
grid_origin[:3,3] = grid_barycenter

# Minimum and maximum depth used for rejecting voxels outside of the cmaera frustrum
depth_min = 0.
depth_max = opt.grid_dim * voxel_size + near_plane
grid_dims = 3 * [opt.grid_dim]

# Resolution of canonical viewing volume in the depth dimension, in number of voxels.
frustrum_depth = 2 * grid_dims[-1]

model = DeepVoxels(lifting_img_dims=proj_image_dims,
                   frustrum_img_dims=proj_image_dims,
                   grid_dims=grid_dims,
                   use_occlusion_net=not opt.no_occlusion_net,
                   num_grid_feats=opt.num_grid_feats,
                   nf0=opt.nf0,
                   img_sidelength=input_image_dims[0])
model.to(device)

# Projection module
projection = ProjectionHelper(projection_intrinsic=proj_intrinsic,
                              lifting_intrinsic=lift_intrinsic,
                              depth_min=depth_min,
                              depth_max=depth_max,
                              projection_image_dims=proj_image_dims,
                              lifting_image_dims=proj_image_dims,
                              grid_dims=grid_dims,
                              voxel_size=voxel_size,
                              device=device,
                              frustrum_depth=frustrum_depth,
                              near_plane=near_plane)

# L1 loss
criterionL1 = nn.L1Loss(reduction='mean').to(device)

# GAN loss
criterionGAN = GANLoss().to(device)
discriminator = PatchDiscriminator(input_nc=3).to(device)

# Optimizers
optimizerD = torch.optim.Adam(discriminator.parameters(), lr=opt.lr)
optimizerG = torch.optim.Adam(model.parameters(), lr=opt.lr)

print("*" * 100)
print("Frustrum depth")
print(frustrum_depth)
print("Near plane")
print(near_plane)
print("Intrinsic")
print(lift_intrinsic)
print("Number of discriminator parameters:")
util.print_network(discriminator)
print("Number of generator parameters:")
util.print_network(model)
print("*" * 100)


def train():
    discriminator.train()
    model.train()

    if opt.checkpoint:
        util.custom_load(model,
                         opt.checkpoint,
                         discriminator)

    # Create the training dataset loader
    train_dataset = NovelViewTriplets(root_dir=opt.data_root,
                                      img_size=input_image_dims,
                                      sampling_pattern=opt.sampling_pattern)
    dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=8)

    # directory name contains some info about hyperparameters.
    dir_name = os.path.join(datetime.datetime.now().strftime('%m_%d'),
                            datetime.datetime.now().strftime('%H-%M-%S_') +
                            (opt.sampling_pattern + '_') +
                            ('%0.2f_l1_weight_' % opt.l1_weight) +
                            ('%d_trgt_' % opt.num_trgt) +
                            '_' + opt.data_root.strip('/').split('/')[-1] +
                            opt.experiment_name)

    log_dir = os.path.join(opt.logging_root, 'logs', dir_name)
    run_dir = os.path.join(opt.logging_root, 'runs', dir_name)

    data_util.cond_mkdir(log_dir)
    data_util.cond_mkdir(run_dir)

    # Save all command line arguments into a txt file in the logging directory for later referene.
    with open(os.path.join(log_dir, "params.txt"), "w") as out_file:
        out_file.write('\n'.join(["%s: %s" % (key, value) for key, value in vars(opt).items()]))

    writer = SummaryWriter(run_dir)

    iter = opt.start_epoch * len(train_dataset)

    print('Begin training...')
    for epoch in range(opt.start_epoch, opt.max_epoch):
        for trgt_views, nearest_view in dataloader:
            backproj_mapping = projection.comp_lifting_idcs(camera_to_world=nearest_view['pose'].squeeze().to(device),
                                                            grid2world=grid_origin)

            proj_mappings = list()
            for i in range(len(trgt_views)):
                proj_mappings.append(projection.compute_proj_idcs(trgt_views[i]['pose'].squeeze().to(device),
                                                                  grid2world=grid_origin))

            if backproj_mapping is None:
                print("Lifting invalid")
                continue
            else:
                lift_volume_idcs, lift_img_coords = backproj_mapping

            if None in proj_mappings:
                print('Projection invalid')
                continue

            proj_frustrum_idcs, proj_grid_coords = list(zip(*proj_mappings))

            outputs, depth_maps = model(nearest_view['gt_rgb'].to(device),
                                        proj_frustrum_idcs, proj_grid_coords,
                                        lift_volume_idcs, lift_img_coords,
                                        writer=writer)

            # Convert the depth maps to metric
            for i in range(len(depth_maps)):
                depth_maps[i] = ((depth_maps[i] + 0.5) * int(
                    np.ceil(np.sqrt(3) * grid_dims[-1])) * voxel_size + near_plane)

            # We don't enforce a loss on the outermost 5 pixels to alleviate boundary errors
            for i in range(len(trgt_views)):
                outputs[i] = outputs[i][:, :, 5:-5, 5:-5]
                trgt_views[i]['gt_rgb'] = trgt_views[i]['gt_rgb'][:, :, 5:-5, 5:-5]

            l1_losses = list()
            for idx in range(len(trgt_views)):
                l1_losses.append(criterionL1(outputs[idx].contiguous().view(-1).float(),
                                             trgt_views[idx]['gt_rgb'].to(device).view(-1).float()))

            losses_d = []
            losses_g = []

            optimizerD.zero_grad()
            optimizerG.zero_grad()

            for idx in range(len(trgt_views)):
                #######
                ## Train Discriminator
                #######
                out_perm = outputs[idx]  # batch, ndf, height, width

                # Fake forward step
                pred_fake = discriminator.forward(
                    out_perm.detach())  # Detach to make sure no gradients go into generator
                loss_d_fake = criterionGAN(pred_fake, False)

                # Real forward step
                real_input = trgt_views[idx]['gt_rgb'].float().to(device)
                pred_real = discriminator.forward(real_input)
                loss_d_real = criterionGAN(pred_real, True)

                # Combined Loss
                losses_d.append((loss_d_fake + loss_d_real) * 0.5)

                #######
                ## Train generator
                #######
                # Try to fake discriminator
                pred_fake = discriminator.forward(out_perm)
                loss_g_gan = criterionGAN(pred_fake, True)

                loss_g_l1 = l1_losses[idx] * opt.l1_weight
                losses_g.append(loss_g_gan + loss_g_l1)

            loss_d = torch.stack(losses_d, dim=0).mean()
            loss_g = torch.stack(losses_g, dim=0).mean()

            loss_d.backward()
            optimizerD.step()
            loss_g.backward()
            optimizerG.step()

            print("Iter %07d   Epoch %03d   loss_gen %0.4f   loss_discrim %0.4f" % (iter, epoch, loss_g, loss_d))

            if not iter % 100:
                # Write tensorboard logs.
                writer.add_image("Depth",
                                 torchvision.utils.make_grid(
                                     [depth_map.squeeze(dim=0).repeat(3, 1, 1) for depth_map in depth_maps],
                                     scale_each=True, normalize=True).cpu().detach().numpy(),
                                 iter)
                writer.add_image("Nearest_neighbors_rgb",
                                 torchvision.utils.make_grid(nearest_view['gt_rgb'], scale_each=True,
                                                             normalize=True).detach().numpy(),
                                 iter)
                output_vs_gt = torch.cat((torch.cat(outputs, dim=0),
                                          torch.cat([i['gt_rgb'].to(device) for i in trgt_views], dim=0)),
                                         dim=0)
                writer.add_image("Output_vs_gt",
                                 torchvision.utils.make_grid(output_vs_gt,
                                                             scale_each=True,
                                                             normalize=True).cpu().detach().numpy(),
                                 iter)

            writer.add_scalar("out_min", outputs[0].min(), iter)
            writer.add_scalar("out_max", outputs[0].max(), iter)

            writer.add_scalar("trgt_min", trgt_views[0]['gt_rgb'].min(), iter)
            writer.add_scalar("trgt_max", trgt_views[0]['gt_rgb'].max(), iter)

            writer.add_scalar("discrim_loss", loss_d, iter)
            writer.add_scalar("gen_loss_total", loss_g, iter)
            writer.add_scalar("gen_loss_l1", loss_g_l1, iter)
            writer.add_scalar("gen_loss_g", loss_g_gan, iter)

            iter += 1

            if iter % 10000 == 0:
                util.custom_save(model,
                                 os.path.join(log_dir, 'model-epoch_%d_iter_%s.pth' % (epoch, iter)),
                                 discriminator)

    util.custom_save(model,
                     os.path.join(log_dir, 'model-epoch_%d_iter_%s.pth' % (epoch, iter)),
                     discriminator)


def test():
    # Create the training dataset loader
    dataset = TestDataset(pose_dir=os.path.join(opt.data_root, 'pose'))

    util.custom_load(model, opt.checkpoint)
    model.eval()

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)

    dir_name = os.path.join(datetime.datetime.now().strftime('%m_%d'),
                            datetime.datetime.now().strftime('%H-%M-%S_') +
                            '_'.join(opt.checkpoint.strip('/').split('/')[-2:]) + '_'
                            + opt.data_root.strip('/').split('/')[-1])

    traj_dir = os.path.join(opt.logging_root, 'test_traj', dir_name)
    depth_dir = os.path.join(traj_dir, 'depth')

    data_util.cond_mkdir(traj_dir)
    data_util.cond_mkdir(depth_dir)

    forward_time = 0.

    print('starting testing...')
    with torch.no_grad():
        iter = 0
        depth_imgs = []
        for trgt_pose in dataloader:
            trgt_pose = trgt_pose.squeeze().to(device)

            start = time.time()
            # compute projection mapping
            proj_mapping = projection.compute_proj_idcs(trgt_pose.squeeze(), grid_origin)
            if proj_mapping is None:  # invalid sample
                print('(invalid sample)')
                continue

            proj_ind_3d, proj_ind_2d = proj_mapping

            # Run through model
            output, depth_maps, = model(None,
                                        [proj_ind_3d], [proj_ind_2d],
                                        None, None,
                                        None)
            end = time.time()
            forward_time += end - start

            output[0] = output[0][:, :, 5:-5, 5:-5]
            print("Iter %d" % iter)

            output_img = np.array(output[0].squeeze().cpu().detach().numpy())
            output_img = output_img.transpose(1, 2, 0)
            output_img += 0.5
            output_img *= 2 ** 16 - 1
            output_img = output_img.round().clip(0, 2 ** 16 - 1)

            depth_img = depth_maps[0].squeeze(0).cpu().detach().numpy()
            depth_img = depth_img.transpose(1, 2, 0)
            depth_imgs.append(depth_img)

            cv2.imwrite(os.path.join(traj_dir, "img_%05d.png" % iter), output_img.astype(np.uint16)[:, :, ::-1])

            iter += 1

        depth_imgs = np.stack(depth_imgs, axis=0)
        depth_imgs = (depth_imgs - np.amin(depth_imgs)) / (np.amax(depth_imgs) - np.amin(depth_imgs))
        depth_imgs *= 2**16 - 1
        depth_imgs = depth_imgs.round()

        for i in range(len(depth_imgs)):
            cv2.imwrite(os.path.join(depth_dir, "img_%05d.png" % i), depth_imgs[i].astype(np.uint16))

    print("Average forward pass time over %d examples is %f"%(iter, forward_time/iter))


def main():
    if opt.train_test == 'train':
        train()
    elif opt.train_test == 'test':
        test()
    else:
        print("Unknown mode.")
        return None


if __name__ == '__main__':
    main()
