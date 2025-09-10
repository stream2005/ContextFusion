import numpy as np
import cv2
import os
# from skimage.io import imsave
import torch
import torch.nn as nn
import torch.nn.functional as F
import kornia
import random
import h5py
import torch.utils.data as Data


def rgb2y(img):
    y = img[0:1, :, :] * 0.299000 + img[1:2, :, :] * 0.587000 + img[2:3, :, :] * 0.114000
    return y


def image_read_cv2(path, mode='RGB'):
    img_BGR = cv2.imread(path).astype('float32')
    assert mode == 'RGB' or mode == 'GRAY' or mode == 'YCrCb', 'mode error'
    if mode == 'RGB':
        img = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2RGB)
    elif mode == 'GRAY':
        img = np.round(cv2.cvtColor(img_BGR, cv2.COLOR_BGR2GRAY))
    elif mode == 'YCrCb':
        img = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2YCrCb)
    return img


def img_save(image, imagename, savepath):
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    # Convert image to uint8 format if it's not already
    if image.dtype != np.uint8:
        image = image.astype(np.uint8)
    # Save the image using OpenCV
    cv2.imwrite(os.path.join(savepath, "{}.png".format(imagename)), image)


class loss_fusion(nn.Module):
    def __init__(self, coeff_int=1, coeff_grad=1):
        super(loss_fusion, self).__init__()
        self.coeff_int = coeff_int
        self.coeff_grad = coeff_grad

    def forward(self, pre, target):
        loss_int = F.l1_loss(pre, target)
        loss_grad = F.l1_loss(kornia.filters.SpatialGradient()(pre), kornia.filters.SpatialGradient()(target))

        loss_total = self.coeff_int * loss_int + self.coeff_grad * loss_grad
        return loss_total


class Transformer():
    def __init__(self, shift_n, rotate_n, flip_n, deform_n):
        self.shift_n = shift_n
        self.rotate_n = rotate_n
        self.flip_n = flip_n
        self.deform_n = deform_n

    def apply(self, x):
        if self.shift_n > 0:
            x_shift = shift_random(x, self.shift_n)
        if self.rotate_n > 0:
            x_rotate = rotate_random(x, self.rotate_n)
        if self.flip_n > 0:
            x_flip = flip_random(x, self.flip_n)
        if self.deform_n > 0:
            x_deform = deform_random(x, self.deform_n)

        if self.shift_n > 0:
            x = torch.cat((x, x_shift), 0)
        if self.rotate_n > 0:
            x = torch.cat((x, x_rotate), 0)
        if self.flip_n > 0:
            x = torch.cat((x, x_flip), 0)
        if self.deform_n > 0:
            x = torch.cat((x, x_deform), 0)
        return x


def shift_random(x, n_trans=5):
    H, W = x.shape[-2], x.shape[-1]
    assert n_trans <= H - 1 and n_trans <= W - 1, 'n_shifts should less than {}'.format(H - 1)
    shifts_row = random.sample(list(np.concatenate([-1 * np.arange(1, H), np.arange(1, H)])), n_trans)
    shifts_col = random.sample(list(np.concatenate([-1 * np.arange(1, W), np.arange(1, W)])), n_trans)
    x = torch.cat([torch.roll(x, shifts=[sx, sy], dims=[-2, -1]).type_as(x) for sx, sy in zip(shifts_row, shifts_col)],
                  dim=0)
    return x


def rotate_random(data, n_trans=5, random_rotate=False):
    if random_rotate:
        theta_list = random.sample(list(np.arange(1, 359)), n_trans)
    else:
        theta_list = np.arange(10, 360, int(360 / n_trans))
    data = torch.cat(
        [kornia.geometry.transform.rotate(data, torch.Tensor([theta]).type_as(data)) for theta in theta_list], dim=0)
    return data


def deform_random(x, n_trans=3, deform_scale=0.05, sigma=5):
    device = x.device
    b, c, h, w = x.shape
    deformations = []
    for _ in range(n_trans):
        noise = torch.randn(2, h, w, device=device)
        noise = kornia.filters.gaussian_blur2d(noise.unsqueeze(0), (sigma * 3, sigma * 3), (sigma, sigma))[0]
        dx = noise[0] * deform_scale * w
        dy = noise[1] * deform_scale * h
        grid = kornia.utils.create_meshgrid(h, w, device=device).squeeze()
        deformed_grid = grid + torch.stack([dx, dy], dim=-1)
        deformed_x = kornia.geometry.transform.remap(
            x,
            deformed_grid[..., 0].unsqueeze(0),
            deformed_grid[..., 1].unsqueeze(0),
            mode='bicubic',
            align_corners=True,
            padding_mode='border'
        )
        deformed_x = torch.clamp(deformed_x, min=0.0, max=1.0)
        deformations.append(deformed_x)
    return torch.cat(deformations, dim=0)


def flip_random(data, n_trans=3):
    assert n_trans <= 3, 'n_flip should less than 3'

    if n_trans >= 1:
        data1 = kornia.geometry.transform.hflip(data)
    if n_trans >= 2:
        data2 = kornia.geometry.transform.vflip(data)
        data1 = torch.cat((data1, data2), 0)
    if n_trans == 3:
        data1 = torch.cat((data1, kornia.geometry.transform.hflip(data2)), 0)
    return data1


class H5Dataset(Data.Dataset):
    def __init__(self, h5file_path):
        self.h5file_path = h5file_path
        h5f = h5py.File(h5file_path, 'r')
        self.keys = list(h5f['ir_patchs'].keys())
        h5f.close()

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        h5f = h5py.File(self.h5file_path, 'r')
        key = self.keys[index]
        IR = np.array(h5f['ir_patchs'][key])
        VIS = np.array(h5f['vis_patchs'][key])
        h5f.close()
        return torch.Tensor(IR), torch.Tensor(VIS), index


class H5Dataset_AiAv(Data.Dataset):
    def __init__(self, h5file_path):
        self.h5file_path = h5file_path
        h5f = h5py.File(h5file_path, 'r')
        self.keys = list(h5f['input_patchs'].keys())
        h5f.close()

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        h5f = h5py.File(self.h5file_path, 'r')
        key = self.keys[index]
        IR = np.array(h5f['input_patchs'][key])
        VIS = np.array(h5f['target_patchs'][key])
        h5f.close()
        return torch.Tensor(IR), torch.Tensor(VIS), index


class ValDataset(Data.Dataset):
    def __init__(self, ir_path, vis_path, mode='RGB'):
        self.ir_path = ir_path
        self.vis_path = vis_path
        self.mode = mode
        self.ir_files = os.listdir(ir_path)
        self.vis_files = os.listdir(vis_path)
        assert len(self.ir_files) == len(self.vis_files), "The number of IR and VIS images must be the same."

    def __len__(self):
        return len(self.ir_files)

    def __getitem__(self, index):
        ir_img_path = os.path.join(self.ir_path, self.ir_files[index])
        vis_img_path = os.path.join(self.vis_path, self.vis_files[index])

        ir_img = np.expand_dims(cv2.imread(ir_img_path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.,
                                axis=-1).transpose(2, 0, 1)  # [1, H, W] Float32
        vis_img = cv2.imread(vis_img_path).astype(np.float32)[:, :, ::-1].transpose(2, 0,
                                                                                    1) / 255.  # [3, H, W] Uint8->float32
        vis_img = rgb2y(vis_img)  # [1, H, W] Float32

        ir_img_tensor = torch.from_numpy(ir_img)
        vis_img_tensor = torch.from_numpy(vis_img)

        return ir_img_tensor, vis_img_tensor
