import argparse
import os
import pickle
import random

import numpy as np
import open3d as o3d
import PIL.Image
import smplx
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from PIL.ImageOps import exif_transpose
from scipy.spatial.transform import Rotation as R
from torch.utils.data import DataLoader, Dataset

from pedscene.utils.rot import (axis_angle_to_matrix, matrix_to_axis_angle,
                                matrix_to_rotation_6d, rotation_6d_to_matrix)


def data_preprocess(label_dict, mode, data_aug_cfg, smpl):
    data_dict = {}

    img_id_1 = label_dict["imgs"][0].split("/")
    img_id_1 = img_id_1[-3] + "_" + img_id_1[-1][:-4]
    img_id_2 = label_dict["imgs"][1].split("/")
    img_id_2 = img_id_2[-3] + "_" + img_id_2[-1][:-4]

    img_1, true_shape_1, scale_factor = preprocess_image(
        label_dict["imgs"][0], 512)
    img_2, true_shape_2, _ = preprocess_image(label_dict["imgs"][1], 512)

    data_dict["imgs"] = torch.stack([img_1, img_2], dim=0)  # 2, C, H, W
    data_dict["img_shapes"] = torch.stack([true_shape_1, true_shape_2],
                                          dim=0)  # 2, 2

    meta = {
        "img_id": [img_id_1, img_id_2],
        "scale_factor": scale_factor,
    }
    data_dict["meta"] = meta

    init_trans = label_dict["trans"][0]
    trans = label_dict["trans"][1] - label_dict["trans"][0]
    init_rot = label_dict["rot"][0]
    rot = torch.einsum("ij, jk->ik", label_dict["rot"][0].T,
                       label_dict["rot"][1])
    trans = init_rot.T @ trans

    data_dict["trans"] = trans  # 3
    data_dict["rot"] = matrix_to_rotation_6d(rot)  # 6
    data_dict["init_trans"] = init_trans  # 3
    data_dict["init_rot"] = matrix_to_rotation_6d(init_rot)  # 6
    data_dict["bbox"] = label_dict["bbox"] * scale_factor  # 2, 4

    data_dict["intrinsics"] = label_dict["intrinsics"] * scale_factor
    data_dict["pose"] = label_dict["pose"]
    data_dict["beta"] = label_dict["beta"]

    smpl_joints_local_1 = smpl[label_dict['gender']](
        transl=label_dict["local_trans"][0].reshape(1, 3),
        global_orient=matrix_to_axis_angle(label_dict["local_rot"][0]).reshape(
            1, 3),
        body_pose=label_dict["pose"][0].reshape(1, 23 * 3),
        betas=label_dict["beta"].unsqueeze(0))

    smpl_joints_local_2 = smpl[label_dict['gender']](
        transl=label_dict["local_trans"][1].reshape(1, 3),
        global_orient=matrix_to_axis_angle(label_dict["local_rot"][1]).reshape(
            1, 3),
        body_pose=label_dict["pose"][1].reshape(1, 23 * 3),
        betas=label_dict["beta"].unsqueeze(0))

    smpl_joints_local_1 = smpl_joints_local_1.joints[0, :24].detach()
    smpl_joints_local_2 = smpl_joints_local_2.joints[0, :24].detach()
    smpl_joints_local_1 = torch.stack([
        smpl_joints_local_1[:, 0] * data_dict["intrinsics"][0] /
        smpl_joints_local_1[:, 2] + data_dict["intrinsics"][2],
        smpl_joints_local_1[:, 1] * data_dict["intrinsics"][1] /
        smpl_joints_local_1[:, 2] + data_dict["intrinsics"][3]
    ],
                                      dim=-1)

    smpl_joints_local_2 = torch.stack([
        smpl_joints_local_2[:, 0] * data_dict["intrinsics"][0] /
        smpl_joints_local_2[:, 2] + data_dict["intrinsics"][2],
        smpl_joints_local_2[:, 1] * data_dict["intrinsics"][1] /
        smpl_joints_local_2[:, 2] + data_dict["intrinsics"][3]
    ],
                                      dim=-1)

    H, W = true_shape_1
    smpl_joints_local_1[smpl_joints_local_1 < 0] = 0
    smpl_joints_local_2[smpl_joints_local_2 < 0] = 0
    smpl_joints_local_1[smpl_joints_local_1[:, 0] > W, 0] = W
    smpl_joints_local_1[smpl_joints_local_1[:, 1] > H, 1] = H
    smpl_joints_local_2[smpl_joints_local_2[:, 0] > W, 0] = W
    smpl_joints_local_2[smpl_joints_local_2[:, 1] > H, 1] = H

    data_dict["joints_2d"] = torch.stack(
        [smpl_joints_local_1, smpl_joints_local_2], dim=0)  # 2, 24, 2

    if mode == "train":
        data_dict = data_aug(data_dict, **data_aug_cfg)

    data_dict["imgs"] = TF.normalize(data_dict["imgs"], (0.5, 0.5, 0.5),
                                     (0.5, 0.5, 0.5))

    smpl_joints_1 = smpl[label_dict['gender']](
        body_pose=data_dict["pose"][0].reshape(1, 23 * 3),
        betas=data_dict["beta"].unsqueeze(0))
    smpl_joints_2 = smpl[label_dict['gender']](
        body_pose=data_dict["pose"][1].reshape(1, 23 * 3),
        betas=data_dict["beta"].unsqueeze(0))

    smpl_joints = torch.concat(
        [smpl_joints_1.joints[:, :24], smpl_joints_2.joints[:, :24]],
        dim=0).detach()

    data_dict["joints"] = smpl_joints  # 2, 24, 3

    return data_dict


def color_jittering(data_dict):
    color_jitter = transforms.ColorJitter(
        brightness=0.2,  # Adjust brightness by a factor of 0.2
        contrast=0.2,  # Adjust contrast by a factor of 0.2
        saturation=0.2,  # Adjust saturation by a factor of 0.2
        hue=0.1  # Adjust hue by a factor of 0.1
    )

    # Since the ColorJitter expects a PIL Image or a tensor of shape (C, H, W), we can directly apply it
    imgs = color_jitter(data_dict['imgs'])
    return imgs


def get_min_scale(img, bbox):
    _, H, W = img.shape
    x1, y1, x2, y2 = bbox
    x1 = int(max(x1, 0))
    y1 = int(max(y1, 0))
    x2 = int(min(x2, W))
    y2 = int(min(y2, H))
    min_scale = max((x2 - x1) / W, (y2 - y1) / H)
    return min_scale


def crop_image(img, bbox, joints2d, scale):

    _, H, W = img.shape
    x1, y1, x2, y2 = bbox
    x1 = int(max(x1, 0))
    y1 = int(max(y1, 0))
    x2 = int(min(x2, W))
    y2 = int(min(y2, H))
    crop_x1 = random.randint(max(int(x2) - int(W * scale), 0),
                             min(int(x1), W - int(W * scale)))
    crop_y1 = random.randint(max(int(y2) - int(H * scale), 0),
                             min(int(y1), H - int(H * scale)))

    cropped_img = img[:, crop_y1:crop_y1 + int(H * scale),
                      crop_x1:crop_x1 + int(W * scale)]
    cropped_img = TF.resize(cropped_img, [H, W])

    scale_y = int(H * scale) / H
    scale_x = int(W * scale) / W

    cropped_bbox = torch.tensor([(x1 - crop_x1) / scale_x,
                                 (y1 - crop_y1) / scale_y,
                                 (x2 - crop_x1) / scale_x,
                                 (y2 - crop_y1) / scale_y])

    joints2d = torch.stack([(joints2d[:, 0] - crop_x1) / scale_x,
                            (joints2d[:, 1] - crop_y1) / scale_y],
                           dim=-1)

    return cropped_img, cropped_bbox, joints2d


def rotate_point(points, angle, W, H, keep_bound=True):
    angle_rad = np.deg2rad(angle)
    rotation_matrix = torch.Tensor([[np.cos(angle_rad), -np.sin(angle_rad)],
                                    [np.sin(angle_rad),
                                     np.cos(angle_rad)]])

    points[..., 0] = points[..., 0] - W // 2
    points[..., 1] = points[..., 1] - H // 2

    rotated_points = torch.einsum("ij, bj -> bi", rotation_matrix, points)

    rotated_points[..., 0] = rotated_points[..., 0] + W // 2
    rotated_points[..., 1] = rotated_points[..., 1] + H // 2

    if keep_bound:
        rotated_points[rotated_points < 0] = 0
        rotated_points[rotated_points[..., 0] > W, 0] = W
        rotated_points[rotated_points[..., 1] > H, 1] = H

    return rotated_points


def rotate_bbox(bbox, angle, W, H):
    x1, y1, x2, y2 = bbox
    bbox_points = torch.Tensor([[x1, y1], [x1, y2], [x2, y1], [x2, y2]])
    new_bbox_points = rotate_point(bbox_points, angle, W, H)
    new_x1 = torch.min(new_bbox_points[:, 0])
    new_x2 = torch.max(new_bbox_points[:, 0])
    new_y1 = torch.min(new_bbox_points[:, 1])
    new_y2 = torch.max(new_bbox_points[:, 1])
    new_bbox = torch.Tensor([new_x1, new_y1, new_x2, new_y2])
    return new_bbox


def data_aug(data_dict,
             random_crop=True,
             color_jitter=False,
             random_flip=False,
             random_rotate=False,
             **kwargs):
    _, H, W = data_dict['imgs'][0].shape
    if random_crop:  # cropping without modify the aspect ratio
        min_scale_1 = get_min_scale(data_dict['imgs'][0], data_dict["bbox"][0])
        min_scale_2 = get_min_scale(data_dict['imgs'][1], data_dict["bbox"][1])
        scale = random.uniform(max(min_scale_1, min_scale_2), 1.0)

        img1, bbox1, joints2d1 = crop_image(data_dict['imgs'][0],
                                            data_dict["bbox"][0],
                                            data_dict["joints_2d"][0], scale)
        img2, bbox2, joints2d2 = crop_image(data_dict['imgs'][1],
                                            data_dict["bbox"][1],
                                            data_dict["joints_2d"][1], scale)

        data_dict['imgs'] = torch.stack([img1, img2], dim=0)
        data_dict["bbox"] = torch.stack([bbox1, bbox2], dim=0)
        data_dict["joints_2d"] = torch.stack([joints2d1, joints2d2], dim=0)

    if color_jitter:
        data_dict['imgs'] = color_jittering(data_dict)

    if random_flip:
        if random.random() > 0.5:
            img1 = TF.hflip(data_dict['imgs'][0])
            img2 = TF.hflip(data_dict['imgs'][1])
            data_dict['imgs'] = torch.stack([img1, img2], dim=0)
            SMPL_JOINTS_FLIP_PERM = [
                2, 1, 3, 5, 4, 6, 8, 7, 9, 11, 10, 12, 14, 13, 15, 17, 16, 19,
                18, 21, 20, 23, 22
            ]
            SMPL_POSE_FLIP_PERM = []
            for i in SMPL_JOINTS_FLIP_PERM:
                SMPL_POSE_FLIP_PERM.append(3 * (i - 1))
                SMPL_POSE_FLIP_PERM.append(3 * (i - 1) + 1)
                SMPL_POSE_FLIP_PERM.append(3 * (i - 1) + 2)

            data_dict["pose"] = data_dict["pose"].reshape(-1, 23 * 3)

            data_dict["pose"] = data_dict["pose"][:, SMPL_POSE_FLIP_PERM]
            data_dict["pose"][:, 1::3] = -data_dict["pose"][:, 1::3]
            data_dict["pose"][:, 2::3] = -data_dict["pose"][:, 2::3]
            data_dict["pose"] = data_dict["pose"].reshape(-1, 23, 3)

            flip_transfrom = torch.eye(4)
            flip_transfrom[0, 0] = -1

            current_transform = torch.eye(4)
            current_transform[:3, :3] = rotation_6d_to_matrix(data_dict["rot"])
            current_transform[:3, 3] = data_dict["trans"]

            current_transform = flip_transfrom @ current_transform @ flip_transfrom

            data_dict["trans"] = current_transform[:3, 3]
            data_dict["rot"] = matrix_to_rotation_6d(current_transform[:3, :3])

            data_dict["joints_2d"][:, :,
                                   0] = W - data_dict["joints_2d"][:, :, 0]
            data_dict["bbox"][:, [0, 2]] = W - data_dict["bbox"][:, [0, 2]]

    if random_rotate:
        angle1 = random.uniform(-5, 5)
        angle2 = random.uniform(-5, 5)
        img1 = TF.rotate(data_dict['imgs'][0], angle1)
        img2 = TF.rotate(data_dict['imgs'][1], angle2)
        data_dict['imgs'] = torch.stack([img1, img2], dim=0)
        joints2d1 = rotate_point(data_dict["joints_2d"][0], -angle1, W, H)
        joints2d2 = rotate_point(data_dict["joints_2d"][1], -angle2, W, H)

        data_dict["joints_2d"] = torch.stack([joints2d1, joints2d2], dim=0)
        bbox1 = rotate_bbox(data_dict["bbox"][0], -angle1, W, H)
        bbox2 = rotate_bbox(data_dict["bbox"][1], -angle2, W, H)
        data_dict["bbox"] = torch.stack([bbox1, bbox2], dim=0)

    return data_dict


def collate_fn(data):
    """Custom collate function."""
    img_batch = []
    img_shape_batch = []
    trans_batch = []
    rot_batch = []
    init_trans_batch = []
    init_rot_batch = []
    bbox_batch = []
    pose_batch = []
    joints_batch = []
    beta_batch = []
    intrinsics_batch = []
    meta_batch = []
    feats_batch = []
    joints_2d_batch = []

    for data_dict in data:
        img_batch.append(data_dict["imgs"])
        img_shape_batch.append(data_dict["img_shapes"])
        trans_batch.append(data_dict["trans"])
        rot_batch.append(data_dict["rot"])
        init_trans_batch.append(data_dict["init_trans"])
        init_rot_batch.append(data_dict["init_rot"])
        bbox_batch.append(data_dict["bbox"])
        pose_batch.append(data_dict["pose"])
        joints_batch.append(data_dict["joints"])
        beta_batch.append(data_dict["beta"])
        intrinsics_batch.append(data_dict["intrinsics"])
        meta_batch.append(data_dict["meta"])
        joints_2d_batch.append(data_dict["joints_2d"])
        if "feats" in data_dict:
            feats_batch.append(data_dict["feats"])

    ret_dict = {
        "imgs": torch.stack(img_batch),
        "img_shapes": torch.stack(img_shape_batch),
        "trans": torch.stack(trans_batch),
        "rot": torch.stack(rot_batch),
        "init_trans": torch.stack(init_trans_batch),
        "init_rot": torch.stack(init_rot_batch),
        "bbox": torch.stack(bbox_batch),
        "batch_size": len(img_batch),
        "pose": torch.stack(pose_batch),
        "joints": torch.stack(joints_batch),
        "beta": torch.stack(beta_batch),
        "meta": meta_batch,
        "intrinsics": torch.stack(intrinsics_batch),
        "joints_2d": torch.stack(joints_2d_batch),
    }

    if len(feats_batch) == ret_dict["batch_size"]:
        ret_dict["feats"] = {}
        ret_dict["feats"]["dec1"] = torch.stack(
            [x["dec1"] for x in feats_batch], dim=0)
        ret_dict["feats"]["dec2"] = torch.stack(
            [x["dec2"] for x in feats_batch], dim=0)
        ret_dict["feats"]["pos1"] = torch.stack(
            [x["pos1"] for x in feats_batch], dim=0)
        ret_dict["feats"]["pos2"] = torch.stack(
            [x["pos2"] for x in feats_batch], dim=0)

    return ret_dict


def preprocess_image(img_name, long_edge_size):

    img = exif_transpose(PIL.Image.open(os.path.join(img_name))).convert("RGB")
    S = max(img.size)
    if S > long_edge_size:
        interp = PIL.Image.LANCZOS
    elif S <= long_edge_size:
        interp = PIL.Image.BICUBIC
    new_size = tuple(int(round(x * long_edge_size / S)) for x in img.size)
    img = img.resize(new_size, interp)
    true_shape = torch.tensor(np.int32(img.size[::-1])).long()
    img = TF.to_tensor(img)

    return img, true_shape, long_edge_size / S


def camera_to_pixel(X, intrinsics, distortion_coefficients):
    # focal length
    f = intrinsics[:2]
    # center principal point
    c = intrinsics[2:]
    k = np.array([
        distortion_coefficients[0], distortion_coefficients[1],
        distortion_coefficients[4]
    ])
    p = np.array([distortion_coefficients[2], distortion_coefficients[3]])
    XX = X[..., :2] / (X[..., 2:])
    # XX = pd.to_numeric(XX, errors='coere')
    r2 = np.sum(XX[..., :2]**2, axis=-1, keepdims=True)

    radial = 1 + np.sum(
        k * np.concatenate((r2, r2**2, r2**3), axis=-1), axis=-1, keepdims=True)

    tan = 2 * np.sum(p * XX[..., ::-1], axis=-1, keepdims=True)
    XXX = XX * (radial + tan) + r2 * p[..., ::-1]
    return f * XXX + c


def world_to_pixels(X, extrinsic_matrix, cam):
    B, N, dim = X.shape
    X = np.concatenate((X, np.ones((B, N, 1))), axis=-1).transpose(0, 2, 1)
    X = (extrinsic_matrix @ X).transpose(0, 2, 1)
    X = camera_to_pixel(X[..., :3].reshape(B * N, dim), cam['intrinsics'],
                        [0] * 5)
    X = X.reshape(B, N, -1)

    def check_pix(p):
        rule1 = p[:, 0] > 0
        rule2 = p[:, 0] < cam['width']
        rule3 = p[:, 1] > 0
        rule4 = p[:, 1] < cam['height']
        rule = [
            a and b and c and d
            for a, b, c, d in zip(rule1, rule2, rule3, rule4)
        ]
        return p[rule] if len(rule) > 50 else []

    X = [check_pix(xx) for xx in X]

    return X


def get_bool_from_coordinates(coordinates, shape=(1080, 1920)):
    bool_arr = np.zeros(shape, dtype=bool)
    if len(coordinates) > 0:
        bool_arr[coordinates[:, 0], coordinates[:, 1]] = True

    return bool_arr


def fix_points_num(points: np.array, num_points: int):
    """
    downsamples the points using voxel and uniform downsampling, 
    and either repeats or randomly selects points to reach the desired number.
    
    Args:
      points (np.array): a numpy array containing 3D points.
      num_points (int): the desired number of points 
    
    Returns:
      a numpy array `(num_points, 3)`
    """
    if len(points) == 0:
        return np.zeros((num_points, 3))
    points = points[~np.isnan(points).any(axis=-1)]

    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(points)
    pc = pc.voxel_down_sample(voxel_size=0.05)
    ratio = int(len(pc.points) / num_points + 0.05)
    if ratio > 1:
        pc = pc.uniform_down_sample(ratio)

    points = np.asarray(pc.points)
    origin_num_points = points.shape[0]

    if origin_num_points < num_points:
        num_whole_repeat = num_points // origin_num_points
        res = points.repeat(num_whole_repeat, axis=0)
        num_remain = num_points % origin_num_points
        res = np.vstack((res, res[:num_remain]))
    else:
        res = points[np.random.choice(origin_num_points, num_points)]
    return res


INTRINSICS = [599.628, 599.466, 971.613, 540.258]
DIST = [0.003, -0.003, -0.001, 0.004, 0.0]
LIDAR2CAM = [[[-0.0355545576, -0.999323133, -0.0094419378, -0.00330376451],
              [0.00117895777, 0.00940596282, -0.999955068, -0.0498469479],
              [0.999367041, -0.0355640917, 0.00084373493, -0.0994979365],
              [0.0, 0.0, 0.0, 1.0]]]


class SLOPER4D_Dataset(Dataset):

    def __init__(self,
                 pkl_file,
                 device='cpu',
                 return_torch: bool = True,
                 fix_pts_num: bool = False,
                 print_info: bool = True,
                 return_smpl: bool = False):

        with open(pkl_file, 'rb') as f:
            data = pickle.load(f)

        self.data = data
        self.pkl_file = pkl_file
        self.device = device
        self.return_torch = return_torch
        self.print_info = print_info
        self.fix_pts_num = fix_pts_num
        self.return_smpl = return_smpl

        self.framerate = data['framerate']  # scalar
        self.length = data['total_frames'] if 'total_frames' in data else len(
            data['frame_num'])

        self.world2lidar, self.lidar_tstamps = self.get_lidar_data()
        self.load_3d_data(data)
        self.load_rgb_data(data)
        self.load_mask(pkl_file)

        self.check_length()

    def get_lidar_data(self, is_inv=True):
        lidar_traj = self.data['first_person']['lidar_traj'].copy()
        lidar_tstamps = lidar_traj[:self.length, -1]
        world2lidar = np.array([np.eye(4)] * self.length)
        world2lidar[:, :3, :3] = R.from_quat(lidar_traj[:self.length,
                                                        4:8]).inv().as_matrix()
        world2lidar[:, :3,
                    3:] = -world2lidar[:, :3, :3] @ lidar_traj[:self.length,
                                                               1:4].reshape(
                                                                   -1, 3, 1)

        return world2lidar, lidar_tstamps

    def load_rgb_data(self, data):
        try:
            self.cam = data['RGB_info']
        except:
            print('=====> Load default camera parameters.')
            self.cam = {
                'fps': 20,
                'width': 1920,
                'height': 1080,
                'intrinsics': INTRINSICS,
                'lidar2cam': LIDAR2CAM,
                'dist': DIST
            }

        if 'RGB_frames' not in data:
            data['RGB_frames'] = {}
            world2lidar, lidar_tstamps = self.get_lidar_data()
            data['RGB_frames']['file_basename'] = [''] * self.length
            data['RGB_frames']['lidar_tstamps'] = lidar_tstamps[:self.length]
            data['RGB_frames']['bbox'] = [[]] * self.length
            data['RGB_frames']['skel_2d'] = [[]] * self.length
            data['RGB_frames']['cam_pose'] = self.cam['lidar2cam'] @ world2lidar
            self.save_pkl(overwrite=True)

        self.file_basename = data['RGB_frames'][
            'file_basename']  # synchronized img file names
        self.lidar_tstamps = data['RGB_frames'][
            'lidar_tstamps']  # synchronized ldiar timestamps
        self.bbox = data['RGB_frames'][
            'bbox']  # 2D bbox of the tracked human (N, [x1, y1, x2, y2])
        self.skel_2d = data['RGB_frames'][
            'skel_2d']  # 2D keypoints (N, [17, 3]), every joint is (x, y, probability)
        self.cam_pose = data['RGB_frames'][
            'cam_pose']  # extrinsic, world to camera (N, [4, 4])

        if self.return_smpl:
            self.smpl_verts, _ = self.return_smpl_verts()
            self.smpl_mask = world_to_pixels(self.smpl_verts, self.cam_pose,
                                             self.cam)

    def load_mask(self, pkl_file):
        mask_pkl = pkl_file[:-4] + "_mask.pkl"
        if os.path.exists(mask_pkl):
            with open(mask_pkl, 'rb') as f:
                print(f'Loading: {mask_pkl}')
                self.masks = pickle.load(f)['masks']
        else:
            self.masks = [[]] * self.length

    def load_3d_data(self, data, person='second_person', points_num=1024):
        assert self.length <= len(
            data['frame_num']
        ), f"RGB length must be less than point cloud length"
        point_clouds = [[]] * self.length
        if 'point_clouds' in data[person]:
            for i, pf in enumerate(data[person]['point_frame']):
                index = data['frame_num'].index(pf)
                if index < self.length:
                    point_clouds[index] = data[person]['point_clouds'][i]
        if self.fix_pts_num:
            point_clouds = np.array(
                [fix_points_num(pts, points_num) for pts in point_clouds])

        sp = data['second_person']
        self.smpl_pose = sp['opt_pose'][:self.length].astype(
            np.float32)  # n x 72 array of scalars
        self.global_trans = sp['opt_trans'][:self.length].astype(
            np.float32)  # n x 3 array of scalars
        self.betas = sp['beta']  # n x 10 array of scalars
        self.smpl_gender = sp['gender']  # male/female/neutral
        self.human_points = point_clouds  # list of n arrays, each of shape (x_i, 3)

    def updata_pkl(self, img_name, bbox=None, cam_pose=None, keypoints=None):
        if img_name in self.file_basename:
            index = self.file_basename.index(img_name)
            if bbox is not None:
                self.data['RGB_frames']['bbox'][index] = bbox
            if keypoints is not None:
                self.data['RGB_frames']['skel_2d'][index] = keypoints
            if cam_pose is not None:
                self.data['RGB_frames']['cam_pose'][index] = cam_pose
        else:
            print(f"{img_name} is not in the synchronized labels file")

    def get_rgb_frames(self,):
        return self.data['RGB_frames']

    def save_pkl(self, overwrite=False):

        save_path = self.pkl_file if overwrite else self.pkl_file[:-4] + '_updated.pkl'
        with open(save_path, 'wb') as f:
            pickle.dump(self.data, f)
        print(f"{save_path} saved")

    def check_length(self):
        # Check if all the lists inside rgb_frames have the same length
        assert all(
            len(lst) == self.length for lst in [
                self.bbox, self.skel_2d, self.lidar_tstamps, self.masks,
                self.smpl_pose, self.global_trans, self.human_points
            ])

        print(f'Data length: {self.length}')

    def get_cam_params(self):
        return torch.from_numpy(np.array(self.cam['lidar2cam']).astype(np.float32)).to(self.device), \
               torch.from_numpy(np.array(self.cam['intrinsics']).astype(np.float32)).to(self.device), \
               torch.from_numpy(np.array(self.cam['dist']).astype(np.float32)).to(self.device)

    def get_img_shape(self):
        return self.cam['width'], self.cam['height']

    def return_smpl_verts(self,):
        file_path = os.path.dirname(os.path.abspath(__file__))
        with torch.no_grad():
            human_model = smplx.create(f"{os.path.dirname(file_path)}/smpl",
                                       gender=self.smpl_gender,
                                       use_face_contour=False,
                                       ext="npz")
            orient = torch.tensor(self.smpl_pose).float()[:, :3]
            bpose = torch.tensor(self.smpl_pose).float()[:, 3:]
            transl = torch.tensor(self.global_trans).float()
            smpl_md = human_model(betas=torch.tensor(self.betas).reshape(
                -1, 10).float(),
                                  return_verts=True,
                                  body_pose=bpose,
                                  global_orient=orient,
                                  transl=transl)

        return smpl_md.vertices.numpy(), smpl_md.joints.numpy()

    def __getitem__(self, index):
        sample = {
            'file_basename':
                self.file_basename[index],  # image file name            
            'lidar_tstamps':
                self.lidar_tstamps[index],  # lidar timestamp           
            'lidar_pose':
                self.world2lidar[
                    index
                ],  # 4*4 transformation, world to lidar                    
            'bbox':
                self.
                bbox[index],  # 2D bbox (x1, y1, x2, y2)                      
            'mask':
                get_bool_from_coordinates(self.masks[index]
                                         ),  # 2D mask (height, width)
            'skel_2d':
                self.skel_2d[
                    index
                ],  # 2D keypoints (x, y, probability)                    
            'cam_pose':
                self.cam_pose[
                    index
                ],  # 4*4 transformation, world to camera                    
            'smpl_pose':
                torch.tensor(self.smpl_pose[index]).float().to(self.device),
            'global_trans':
                torch.tensor(self.global_trans[index]).float().to(self.device),
            'betas':
                torch.tensor(self.betas).float().to(self.device),

            # 2D mask of SMPL on images, (n, [x, y]), where (x, y) is the pixel coordinate on the image
            'smpl_mask':
                self.smpl_mask[index] if hasattr(self, 'smpl_mask') else [],
            'smpl_verts':
                self.smpl_verts[index] if hasattr(self, 'smpl_verts') else [],

            # in world coordinates, (n, (x, y, z)), the n is different in each frame
            # if fix_point_num is True, the every frame will be resampled to 1024 points
            'human_points':
                self.human_points[index],
        }

        if self.return_torch:
            for k, v in sample.items():
                if k == 'mask':
                    sample[k] = torch.tensor(v).to(self.device)
                elif type(v) != str and type(v) != torch.Tensor:
                    sample[k] = torch.tensor(v).float().to(self.device)

        mispart = ''
        mispart += 'box ' if len(sample['bbox']) < 1 else ''
        mispart += 'kpt ' if len(sample['skel_2d']) < 1 else ''
        mispart += 'pts ' if len(sample['human_points']) < 1 else ''

        if len(mispart) > 0 and self.print_info:
            print(f'Missing {mispart} in: {index} ')

        return sample

    def __len__(self):
        return self.length


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SLOPER4D dataset')
    parser.add_argument(
        '--pkl_file',
        type=str,
        default=
        '/wd8t/sloper4d_publish/seq003_street_002/seq003_street_002_labels.pkl',
        help='Path to the pkl file')
    parser.add_argument('--batch_size',
                        type=int,
                        default=1,
                        help='The batch size of the data loader')
    parser.add_argument('--index',
                        type=int,
                        default=-1,
                        help='the index frame to be saved to a image')
    args = parser.parse_args()

    dataset = SLOPER4D_Dataset(args.pkl_file,
                               return_torch=False,
                               fix_pts_num=True)

    # =====> attention
    # Batch_size > 1 is not supported yet
    # because bbox and 2d keypoints missing in some frames
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    root_folder = os.path.dirname(args.pkl_file)

    for index, sample in enumerate(dataloader):
        for i in range(args.batch_size):
            pcd_name = f"{sample['lidar_tstamps'][i]:.03f}".replace(
                '.', '_') + '.pcd'
            img_path = os.path.join(root_folder, 'rgb_data',
                                    sample['file_basename'][i])
            pcd_path = os.path.join(root_folder, 'lidar_data',
                                    'lidar_frames_rot', pcd_name)
            extrinsic = sample['cam_pose'][
                i]  # 4x4 lidar to camera transformation
            keypoints = sample['skel_2d']  # 2D keypoints, coco17 style
            if index == args.index:
                print(f"{index} pcd path: {pcd_path}")
                print(f"{index} img path: {img_path}")
