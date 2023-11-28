import copy
import pickle
import os
from pathlib import Path

import numpy as np

from pcdet.ops.roiaware_pool3d import roiaware_pool3d_utils
from pcdet.utils import box_utils, common_utils
from pcdet.datasets.dataset import DatasetTemplate
from tqdm import tqdm
import open3d as o3d 
class OwnWZDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None, data_split_type = 'train'):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        # self.split = self.dataset_cfg.DATA_SPLIT[self.mode]
        self.split = self.dataset_cfg.DATA_SPLIT[data_split_type]
        self.label_split_path = self.root_path / ('trainval' if self.split != 'test' else 'test')
        self.point_cloud_split_path = Path(self.dataset_cfg.POINT_CLOUD_ROOT_PATH) / ('trainval' if self.split != 'test' else 'test')

        split_dir = os.path.join(self.root_path, 'ImageSets', (self.split + '.txt'))
        self.sample_id_list = [x.strip() for x in open(split_dir).readlines()] if os.path.exists(split_dir) else None

        self.ownwz_infos = []
        # self.include_data(self.mode)
        self.include_data(data_split_type)
        self.map_class_to_kitti = self.dataset_cfg.MAP_CLASS_TO_KITTI

    def include_data(self, mode):
        if self.logger is not None:
            self.logger.info('Loading Custom dataset.')
        ownwz_infos = []

        for info_path in self.dataset_cfg.INFO_PATH[mode]:
            info_path = self.root_path / info_path
            if not info_path.exists():
                continue
            with open(info_path, 'rb') as f:
                infos = pickle.load(f)
                ownwz_infos.extend(infos)

        self.ownwz_infos.extend(ownwz_infos)
        self.logger.info('Total samples for CUSTOM dataset: %d' % (len(ownwz_infos)))

    def get_label(self, idx):
        label_file = self.label_split_path / 'labels' / ('%s.txt' % idx)
        assert label_file.exists()
        with open(label_file, 'r') as f:
            lines = f.readlines()

        # [N, 8]: (x y z dx dy dz heading_angle category_id)
        gt_boxes = []
        gt_names = []
        for line in lines:
            line_list = line.strip().split(' ')
            gt_boxes.append(line_list[:-1])
            gt_names.append(line_list[-1])

        return np.array(gt_boxes, dtype=np.float32), np.array(gt_names)

    def get_lidar(self, idx):
        lidar_file = self.root_path / 'points' / ('%s.npy' % idx)
        assert lidar_file.exists()
        point_features = np.load(lidar_file)
        return point_features
    
    def get_lidar_bin(self, idx):
        lidar_file = self.root_path / 'bins' / ('%s.bin' % idx)
        assert lidar_file.exists()
        point_features = np.fromfile(lidar_file, dtype=np.float32).reshape(-1, 4)
        return point_features  
      
    def get_lidar_pcd(self, idx):
        lidar_file = self.point_cloud_split_path / 'pcd' / ('%s.pcd' % idx)
        assert lidar_file.exists()
        # point_features = np.fromfile(lidar_file, dtype=np.float32).reshape(-1, 4)
        cloud = o3d.t.io.read_point_cloud(str(lidar_file))
        positions = (cloud.point.positions.numpy())
        intensity = (cloud.point.intensity.numpy() / 255.0)
        # timestamp = (cloud.point.timestamp.numpy())
        # timestamp = np.zeros_like(intensity)
        point_features = np.concatenate((positions,intensity),axis=1)
        point_features = np.array(point_features).astype(float)
        # print("shape:",point_features.shape)
        return point_features

    def set_split(self, split):
        super().__init__(
            dataset_cfg=self.dataset_cfg, class_names=self.class_names, training=self.training,
            root_path=self.root_path, logger=self.logger
        )
        self.split = split
        self.label_split_path = self.root_path / ('trainval' if self.split != 'test' else 'test')
        self.point_cloud_split_path = Path(self.dataset_cfg.POINT_CLOUD_ROOT_PATH) / ('trainval' if self.split != 'test' else 'test')

        split_dir = self.root_path / 'ImageSets' / (self.split + '.txt')
        self.sample_id_list = [x.strip() for x in open(split_dir).readlines()] if split_dir.exists() else None

    def __len__(self):
        if self._merge_all_iters_to_one_epoch:
            return len(self.sample_id_list) * self.total_epochs

        return len(self.ownwz_infos)

    def __getitem__(self, index):
        if self._merge_all_iters_to_one_epoch:
            index = index % len(self.ownwz_infos)

        info = copy.deepcopy(self.ownwz_infos[index])
        sample_idx = info['point_cloud']['lidar_idx']
        points = self.get_lidar_pcd(sample_idx)
        input_dict = {
            'frame_id': self.sample_id_list[index],
            'points': points
        }

        if 'annos' in info:
            annos = info['annos']
            annos = common_utils.drop_info_with_name(annos, name='DontCare')
            gt_names = annos['name']
            gt_boxes_lidar = annos['gt_boxes_lidar']
            input_dict.update({
                'gt_names': gt_names,
                'gt_boxes': gt_boxes_lidar
            })

        data_dict = self.prepare_data(data_dict=input_dict)

        return data_dict

    def evaluation(self, det_annos, class_names, **kwargs):
        if 'annos' not in self.ownwz_infos[0].keys():
            return 'No ground-truth boxes for evaluation', {}

        def kitti_eval(eval_det_annos, eval_gt_annos, map_name_to_kitti):
            from ..kitti.kitti_object_eval_python import eval as kitti_eval
            from ..kitti import kitti_utils

            kitti_utils.transform_annotations_to_kitti_format(eval_det_annos, map_name_to_kitti=map_name_to_kitti)
            kitti_utils.transform_annotations_to_kitti_format(
                eval_gt_annos, map_name_to_kitti=map_name_to_kitti,
                info_with_fakelidar=self.dataset_cfg.get('INFO_WITH_FAKELIDAR', False)
            )
            kitti_class_names = [map_name_to_kitti[x] for x in class_names]
            ap_result_str, ap_dict = kitti_eval.get_official_eval_result(
                gt_annos=eval_gt_annos, dt_annos=eval_det_annos, current_classes=kitti_class_names
            )
            return ap_result_str, ap_dict
        def waymo_eval(eval_det_annos, eval_gt_annos):
            from ..waymo.waymo_eval import OpenPCDetWaymoDetectionMetricsEstimator
            eval = OpenPCDetWaymoDetectionMetricsEstimator()

            ap_dict = eval.waymo_evaluation(
                eval_det_annos, eval_gt_annos, class_name=class_names,
                distance_thresh=50, fake_gt_infos=self.dataset_cfg.get('INFO_WITH_FAKELIDAR', False)
            )
            ap_result_str = '\n'
            for key in ap_dict:
                ap_dict[key] = ap_dict[key][0]
                ap_result_str += '%s: %.4f \n' % (key, ap_dict[key])

            return ap_result_str, ap_dict
        
        eval_det_annos = copy.deepcopy(det_annos)
        eval_gt_annos = [copy.deepcopy(info['annos']) for info in self.ownwz_infos]

        if kwargs['eval_metric'] == 'kitti':
            ap_result_str, ap_dict = kitti_eval(eval_det_annos, eval_gt_annos, self.map_class_to_kitti)
        elif kwargs['eval_metric'] == 'waymo':
            ap_result_str, ap_dict = waymo_eval(eval_det_annos, eval_gt_annos)
        else:
            raise NotImplementedError

        return ap_result_str, ap_dict

    def get_infos(self, class_names, num_workers=4, has_label=True, count_inside_pts=True,sample_id_list=None, num_features=4):
        import concurrent.futures as futures

        def process_single_scene(sample_idx):
            # print('%s sample_idx: %s' % (self.split, sample_idx))
            info = {}
            pc_info = {'num_features': num_features, 'lidar_idx': sample_idx}
            info['point_cloud'] = pc_info

            if has_label:
                annotations = {}
                gt_boxes_lidar, name = self.get_label(sample_idx)
                if name.size == 0:
                    print(sample_idx)
                    return info 
                annotations['name'] = name
                annotations['gt_boxes_lidar'] = gt_boxes_lidar[:, :7]
                annotations['difficulty'] = np.zeros(len(gt_boxes_lidar), np.int32) - 1
                distance = np.linalg.norm(gt_boxes_lidar[:,:3])
                # info['annos'] = annotations
                
                if count_inside_pts:
                    import torch
                    points = self.get_lidar_pcd(sample_idx)
                    point_indices = roiaware_pool3d_utils.points_in_boxes_gpu(
                        torch.from_numpy(points[:, 0:3]).unsqueeze(dim=0).float().cuda(),
                        torch.from_numpy(gt_boxes_lidar[:, 0:7]).unsqueeze(dim=0).float().cuda()
                    ).long().squeeze(dim=0).cpu().numpy()  # (nboxes, npoints)
                    
                    num_points_in_gt = -np.ones(len(gt_boxes_lidar), dtype=np.int32)
                    for i in range(len(gt_boxes_lidar)):
                        gt_points = points[point_indices == i]

                        gt_points[:, :3] -= gt_boxes_lidar[i, :3]
                        num_points_in_gt[i] = gt_points.shape[0]
                    annotations['num_points_in_gt'] = num_points_in_gt
                info['annos'] = annotations


            return info

        sample_id_list = sample_id_list if sample_id_list is not None else self.sample_id_list

        # create a thread pool to improve the velocity
        with futures.ThreadPoolExecutor(num_workers) as executor:
            infos = executor.map(process_single_scene, sample_id_list)
        return list(infos)

    def create_groundtruth_database(self, info_path=None, used_classes=None, split='train'):
        import torch

        database_save_path = Path(self.root_path) / ('gt_database' if split == 'train' else ('gt_database_%s' % split))
        db_info_save_path = Path(self.root_path) / ('ownwz_dbinfos_%s.pkl' % split)

        database_save_path.mkdir(parents=True, exist_ok=True)
        all_db_infos = {}

        with open(info_path, 'rb') as f:
            infos = pickle.load(f)

        # for k in range(len(infos)):
        for k in tqdm(range(len(infos)), desc='gt_db creating'):
            # print('gt_database sample: %d/%d' % (k + 1, len(infos)))
            info = infos[k]
            sample_idx = info['point_cloud']['lidar_idx']
            points = self.get_lidar_pcd(sample_idx)
            annos = info['annos']
            names = annos['name']
            gt_boxes = annos['gt_boxes_lidar']

            num_obj = gt_boxes.shape[0]
            point_indices = roiaware_pool3d_utils.points_in_boxes_gpu(
                torch.from_numpy(points[:, 0:3]).unsqueeze(dim=0).float().cuda(),
                torch.from_numpy(gt_boxes[:, 0:7]).unsqueeze(dim=0).float().cuda()
            ).long().squeeze(dim=0).cpu().numpy()  # (nboxes, npoints)

            for i in range(num_obj):
                filename = '%s_%s_%d.bin' % (sample_idx, names[i], i)
                filepath = database_save_path / filename
                gt_points = points[point_indices == i]

                gt_points[:, :3] -= gt_boxes[i, :3]
                with open(filepath, 'w') as f:
                    gt_points.tofile(f)

                if (used_classes is None) or names[i] in used_classes:
                    db_path = str(filepath.relative_to(self.root_path))  # gt_database/xxxxx.bin
                    db_info = {'name': names[i], 'path': db_path, 'gt_idx': i,
                               'box3d_lidar': gt_boxes[i], 'num_points_in_gt': gt_points.shape[0]}
                    if names[i] in all_db_infos:
                        all_db_infos[names[i]].append(db_info)
                    else:
                        all_db_infos[names[i]] = [db_info]

        # Output the num of all classes in database
        for k, v in all_db_infos.items():
            print('Database %s: %d' % (k, len(v)))

        with open(db_info_save_path, 'wb') as f:
            pickle.dump(all_db_infos, f)

    @staticmethod
    def create_label_file_with_name_and_box(class_names, gt_names, gt_boxes, save_label_path):
        with open(save_label_path, 'w') as f:
            for idx in range(gt_boxes.shape[0]):
                boxes = gt_boxes[idx]
                name = gt_names[idx]
                if name not in class_names:
                    continue
                line = "{x} {y} {z} {l} {w} {h} {angle} {name}\n".format(
                    x=boxes[0], y=boxes[1], z=(boxes[2]), l=boxes[3],
                    w=boxes[4], h=boxes[5], angle=boxes[6], name=name
                )
                f.write(line)


def create_ownwz_infos(dataset_cfg, class_names, data_path, save_path, workers=4):
    dataset = OwnWZDataset(
        dataset_cfg=dataset_cfg, class_names=class_names, root_path=data_path,
        training=False, logger=common_utils.create_logger()
    )
    train_split, val_split, test_split = 'train', 'val', 'test'
    num_features = len(dataset_cfg.POINT_FEATURE_ENCODING.src_feature_list)

    train_filename = save_path / ('ownwz_infos_%s.pkl' % train_split)
    val_filename = save_path / ('ownwz_infos_%s.pkl' % val_split)
    test_filename = save_path / ('ownwz_infos_%s.pkl' % test_split)

    print('------------------------Start to generate data infos------------------------')

    dataset.set_split(train_split)
    ownwz_infos_train = dataset.get_infos(
        class_names, num_workers=workers, has_label=True, count_inside_pts=False, num_features=num_features)
    with open(train_filename, 'wb') as f:
        pickle.dump(ownwz_infos_train, f)
    print('Custom info train file is saved to %s' % train_filename)

    dataset.set_split(val_split)
    ownwz_infos_val = dataset.get_infos(
        class_names, num_workers=workers, has_label=True,count_inside_pts=True, num_features=num_features)
    with open(val_filename, 'wb') as f:
        pickle.dump(ownwz_infos_val, f)
    print('Custom info %s file is saved to %s' % (val_split,val_filename))

    dataset.set_split(test_split)
    ownwz_infos_test = dataset.get_infos(
        class_names, num_workers=workers, has_label=True,count_inside_pts=True, num_features=num_features)
    with open(test_filename, 'wb') as f:
        pickle.dump(ownwz_infos_test, f)
    print('Custom info %s file is saved to %s' % (test_split,test_filename))

    print('------------------------Start create groundtruth database for data augmentation------------------------')
    dataset.set_split(train_split)
    dataset.create_groundtruth_database(train_filename, split=train_split)
    print('------------------------Data preparation done------------------------')


if __name__ == '__main__':
    import sys

    if sys.argv.__len__() > 1 and sys.argv[1] == 'create_ownwz_infos':
        import yaml
        from pathlib import Path
        from easydict import EasyDict

        dataset_cfg = EasyDict(yaml.safe_load(open(sys.argv[2])))
        ROOT_DIR = (Path(__file__).resolve().parent / '../../../').resolve()
        create_ownwz_infos(
            dataset_cfg=dataset_cfg,
            class_names=['Vehicle','BigVehicle', 'Barrier', 'Pedestrian','TrafficCone'],
            # data_path=ROOT_DIR / 'data' / 'kitti',
            # save_path=ROOT_DIR / 'data' / 'kitti'
            data_path=Path(dataset_cfg.DATA_PATH),
            save_path=Path(dataset_cfg.DATA_PATH)
        )
