import os
import argparse
import re
import glob
import onnx
import argparse
import numpy as np
import datetime
from pathlib import Path
import time

import copy

import open3d
from visual_utils import open3d_vis_utils as V

import torch
import torch.nn as nn
from tensorboardX import SummaryWriter

import torch
from torchinfo import summary

from eval_utils import eval_utils

from pcdet.utils import common_utils
from pcdet.models import build_network, load_data_to_gpu, model_fn_decorator
from pcdet.datasets import DatasetTemplate, build_dataloader
from pcdet.config import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file

from train_utils.optimization import build_optimizer, build_scheduler
from train_utils.train_utils import train_model

# 重定义的BaseBEVBackbone
from onnx_utils_dual_radar.onnx_backbone_2d import BaseBEVBackbone

# import pruning.torch_pruning_v0 as tp
import torch_pruning as tp

class DemoDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None, ext='.bin'):
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
        self.root_path = root_path
        self.ext = ext
        data_file_list = glob.glob(str(root_path / f'*{self.ext}')) if self.root_path.is_dir() else [self.root_path]

        data_file_list.sort()
        self.sample_file_list = data_file_list

    def __len__(self):
        return len(self.sample_file_list)

    def __getitem__(self, index):
        if self.ext == '.bin':
            points = np.fromfile(self.sample_file_list[index], dtype=np.float32).reshape(-1, 4)
        elif self.ext == '.npy':
            points = np.load(self.sample_file_list[index])
        else:
            raise NotImplementedError

        input_dict = {
            'points': points,
            'frame_id': index,
        }

        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict


def eval_single_ckpt(model, test_loader, args, eval_output_dir, logger, epoch_id, dist_test=False):    
    # start evaluation
    eval_utils.eval_one_epoch(cfg, args, model, test_loader, epoch_id, logger, dist_test=dist_test,
        result_dir=eval_output_dir
    )
    
    
def parse_config():
    data_path="/root/autodl-tmp/code/OpenPCDet/data/kitti/testing/velodyne/"
    cfg_ = "cfgs/kitti_models/pointpillar.yaml"
    ckpt_ = "ckpt/kitti/pointpillar_7728.pth"

    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default=cfg_,
                        help='specify the config for demo')
    parser.add_argument('--data_path', type=str, default=data_path,
                        help='specify the point cloud data file or directory')
    parser.add_argument('--ckpt', type=str, default=ckpt_,
                        help='specify the pretrained model')
    parser.add_argument('--ext', type=str, default='.bin',
                        help='specify the extension of your point cloud data file')

    parser.add_argument('--batch_size', type=int, default=None, required=False, help='batch size for training')
    parser.add_argument('--workers', type=int, default=4, help='number of workers for dataloader')
    parser.add_argument('--extra_tag', type=str, default='default', help='extra tag for this experiment')
    # parser.add_argument('--ckpt', type=str, default=None, help='checkpoint to start from')
    parser.add_argument('--pretrained_model', type=str, default=None, help='pretrained_model')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm'], default='none')
    parser.add_argument('--tcp_port', type=int, default=18888, help='tcp port for distrbuted training')
    parser.add_argument('--local_rank', type=int, default=0, help='local rank for distributed training')
    parser.add_argument('--set', dest='set_cfgs', default=None, nargs=argparse.REMAINDER,
                        help='set extra config keys if needed')

    parser.add_argument('--max_waiting_mins', type=int, default=30, help='max waiting minutes')
    parser.add_argument('--start_epoch', type=int, default=0, help='')
    parser.add_argument('--eval_tag', type=str, default='default', help='eval tag for this experiment')
    parser.add_argument('--eval_all', action='store_true', default=False, help='whether to evaluate all checkpoints')
    parser.add_argument('--ckpt_dir', type=str, default=None, help='specify a ckpt directory to be evaluated if needed')
    parser.add_argument('--save_to_file', action='store_true', default=False, help='')
    parser.add_argument('--infer_time', action='store_true', default=False, help='calculate inference latency')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)

    return args, cfg


def re_train_parse_config():
    
    cfg_file = "cfgs/kitti_models/pointpillar.yaml"
    
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default=cfg_file, help='specify the config for training')

    parser.add_argument('--batch_size', type=int, default=None, required=False, help='batch size for training')
    parser.add_argument('--epochs', type=int, default=None, required=False, help='number of epochs to train for')
    parser.add_argument('--workers', type=int, default=4, help='number of workers for dataloader')
    parser.add_argument('--extra_tag', type=str, default='default', help='extra tag for this experiment')
    parser.add_argument('--ckpt', type=str, default=None, help='checkpoint to start from')
    parser.add_argument('--pretrained_model', type=str, default=None, help='pretrained_model')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm'], default='none')
    parser.add_argument('--tcp_port', type=int, default=18888, help='tcp port for distrbuted training')
    parser.add_argument('--sync_bn', action='store_true', default=False, help='whether to use sync bn')
    parser.add_argument('--fix_random_seed', action='store_true', default=False, help='')
    parser.add_argument('--ckpt_save_interval', type=int, default=1, help='number of training epochs')
    parser.add_argument('--local_rank', type=int, default=0, help='local rank for distributed training')
    parser.add_argument('--max_ckpt_save_num', type=int, default=30, help='max number of saved checkpoint')
    parser.add_argument('--merge_all_iters_to_one_epoch', action='store_true', default=False, help='')
    parser.add_argument('--set', dest='set_cfgs', default=None, nargs=argparse.REMAINDER,
                        help='set extra config keys if needed')

    parser.add_argument('--max_waiting_mins', type=int, default=0, help='max waiting minutes')
    parser.add_argument('--start_epoch', type=int, default=0, help='')
    parser.add_argument('--num_epochs_to_eval', type=int, default=0, help='number of checkpoints to be evaluated')
    parser.add_argument('--save_to_file', action='store_true', default=False, help='')
    
    parser.add_argument('--use_tqdm_to_record', action='store_true', default=False, help='if True, the intermediate losses will not be logged to file, only tqdm will be used')
    parser.add_argument('--logger_iter_interval', type=int, default=50, help='')
    parser.add_argument('--ckpt_save_time_interval', type=int, default=300, help='in terms of seconds')
    parser.add_argument('--wo_gpu_stat', action='store_true', help='')
    parser.add_argument('--use_amp', action='store_true', help='use mix precision training')
    

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg.TAG = Path(args.cfg_file).stem
    cfg.EXP_GROUP_PATH = '/'.join(args.cfg_file.split('/')[1:-1])  # remove 'cfgs' and 'xxxx.yaml'
    
    args.use_amp = args.use_amp or cfg.OPTIMIZATION.get('USE_AMP', False)

    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs, cfg)

    return args, cfg


def re_trainModel(model):
    args, cfg = re_train_parse_config()
    if args.launcher == 'none':
        dist_train = False
        total_gpus = 1
    else:
        total_gpus, cfg.LOCAL_RANK = getattr(common_utils, 'init_dist_%s' % args.launcher)(
            args.tcp_port, args.local_rank, backend='nccl'
        )
        dist_train = True

    if args.batch_size is None:
        args.batch_size = cfg.OPTIMIZATION.BATCH_SIZE_PER_GPU
    else:
        assert args.batch_size % total_gpus == 0, 'Batch size should match the number of gpus'
        args.batch_size = args.batch_size // total_gpus

    args.epochs = cfg.OPTIMIZATION.NUM_EPOCHS if args.epochs is None else args.epochs

    if args.fix_random_seed:
        common_utils.set_random_seed(666 + cfg.LOCAL_RANK)

    output_dir = cfg.ROOT_DIR / 'output' / cfg.EXP_GROUP_PATH / cfg.TAG / args.extra_tag
    ckpt_dir = output_dir / 'ckpt'
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    log_file = output_dir / ('train_%s.log' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    logger = common_utils.create_logger(log_file, rank=cfg.LOCAL_RANK)

    # log to file
    logger.info('**********************Start logging**********************')
    gpu_list = os.environ['CUDA_VISIBLE_DEVICES'] if 'CUDA_VISIBLE_DEVICES' in os.environ.keys() else 'ALL'
    logger.info('CUDA_VISIBLE_DEVICES=%s' % gpu_list)

    if dist_train:
        logger.info('Training in distributed mode : total_batch_size: %d' % (total_gpus * args.batch_size))
    else:
        logger.info('Training with a single process')
        
    for key, val in vars(args).items():
        logger.info('{:16} {}'.format(key, val))
    log_config_to_file(cfg, logger=logger)
    if cfg.LOCAL_RANK == 0:
        os.system('cp %s %s' % (args.cfg_file, output_dir))

    tb_log = SummaryWriter(log_dir=str(output_dir / 'tensorboard')) if cfg.LOCAL_RANK == 0 else None


    logger.info("----------- Create dataloader & network & optimizer -----------")
    train_set, train_loader, train_sampler = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=args.batch_size,
        dist=False, workers=args.workers,
        logger=logger,
        training=True,
        merge_all_iters_to_one_epoch=args.merge_all_iters_to_one_epoch,
        total_epochs=args.epochs,
        seed=666 if args.fix_random_seed else None
    )

    optimizer = build_optimizer(model, cfg.OPTIMIZATION)

    # load checkpoint if it is possible
    start_epoch = it = 0
    last_epoch = -1
        
    ckpt_list = glob.glob(str(ckpt_dir / '*.pth'))
            
    if len(ckpt_list) > 0:
        ckpt_list.sort(key=os.path.getmtime)
        while len(ckpt_list) > 0:
            try:
                it, start_epoch = model.load_params_with_optimizer(
                    ckpt_list[-1], to_cpu=dist_train, optimizer=optimizer, logger=logger
                )
                last_epoch = start_epoch + 1
                break
            except:
                ckpt_list = ckpt_list[:-1]

    model.train()  # before wrap to DistributedDataParallel to support fixed some parameters
    
    if dist_train:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[cfg.LOCAL_RANK % torch.cuda.device_count()])
    logger.info(f'----------- Model {cfg.MODEL.NAME} created, param count: {sum([m.numel() for m in model.parameters()])} -----------')
    # logger.info(model)

    lr_scheduler, lr_warmup_scheduler = build_scheduler(
        optimizer, total_iters_each_epoch=len(train_loader), total_epochs=args.epochs,
        last_epoch=last_epoch, optim_cfg=cfg.OPTIMIZATION
    )

    # -----------------------start training---------------------------
    logger.info('**********************Start training %s/%s(%s)**********************'
                % (cfg.EXP_GROUP_PATH, cfg.TAG, args.extra_tag))

    train_model(
        model,
        optimizer,
        train_loader,
        model_func=model_fn_decorator(),
        lr_scheduler=lr_scheduler,
        optim_cfg=cfg.OPTIMIZATION,
        start_epoch=start_epoch,
        total_epochs=args.epochs,
        start_iter=it,
        rank=cfg.LOCAL_RANK,
        tb_log=tb_log,
        ckpt_save_dir=ckpt_dir,
        train_sampler=train_sampler,
        lr_warmup_scheduler=lr_warmup_scheduler,
        ckpt_save_interval=args.ckpt_save_interval,
        max_ckpt_save_num=args.max_ckpt_save_num,
        merge_all_iters_to_one_epoch=args.merge_all_iters_to_one_epoch, 
        logger=logger, 
        logger_iter_interval=args.logger_iter_interval,
        ckpt_save_time_interval=args.ckpt_save_time_interval,
        use_logger_to_record=not args.use_tqdm_to_record, 
        show_gpu_stat=not args.wo_gpu_stat,
        use_amp=args.use_amp,
        cfg=cfg
    )

    if hasattr(train_set, 'use_shared_memory') and train_set.use_shared_memory:
        train_set.clean_shared_memory()

    logger.info('**********************End training %s/%s(%s)**********************\n\n\n'
                % (cfg.EXP_GROUP_PATH, cfg.TAG, args.extra_tag))
    
    
def eval_model(model, cfg, args, logger):    
    test_set, test_loader, sampler = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=4,
        dist=False, workers=args.workers, logger=logger, training=False
    )
    
    eval_output_dir =Path('./')
    eval_single_ckpt(model, test_loader, args, eval_output_dir, logger, 1)

    
if __name__ == "__main__":
    args, cfg = parse_config()
    logger = common_utils.create_logger()
    logger.info('-----------------Quick Demo of OpenPCDet-------------------------')
    demo_dataset = DemoDataset(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
        root_path=Path(args.data_path), ext=args.ext, logger=logger
    )   
    logger.info(f'Total number of samples: \t{len(demo_dataset)}')
    
    # 定义新的BaseBEVBackbone
    myModel_BaseBEVBackbone = BaseBEVBackbone(cfg.MODEL.BACKBONE_2D, 64)
    myModel_BaseBEVBackbone.cuda()
    example_inputs = torch.randn(1,64,432,496).cuda()
    summary(myModel_BaseBEVBackbone, input_data=example_inputs)
    
    # 加载模型权重
    checkpoint = torch.load(args.ckpt, map_location='cuda')
    dicts = {}
    for key in checkpoint["model_state"].keys():
        if "backbone_2d" in key:
            dicts[key[12:]] = checkpoint["model_state"][key] # 此处key[12:]表示是去掉backbone_2d字符串
    myModel_BaseBEVBackbone.load_state_dict(dicts)
    
    # —————————————— 以下是剪枝模块代码 —————————————— # 
    # 对BaseBEVBackbone剪枝
    # step 1: 定义重要性评估标准
    imp = tp.importance.GroupNormImportance()
    
    # 摘除不进行剪枝的结构
    ignored_layers = []
    for m in myModel_BaseBEVBackbone.modules():
        if isinstance(m, torch.nn.ConvTranspose2d):
            ignored_layers.append(m)
            
    # print("ignored_layers: ", ignored_layers)
    
    # 定义剪枝器
    iterative_steps = 5
    pruner = tp.pruner.GroupNormPruner(
        myModel_BaseBEVBackbone,
        example_inputs, # 用于分析依赖的伪输入
        importance=imp, # 重要性评估指标
        iterative_steps=iterative_steps, # 迭代剪枝，设为1则一次性完成剪枝
        pruning_ratio=0.2, # 目标稀疏性
        ignored_layers=ignored_layers
    )
    
    # 执行剪枝
    base_macs, base_nparams = tp.utils.count_ops_and_params(myModel_BaseBEVBackbone, example_inputs)
    for i in range(iterative_steps):
        pruner.step() # 执行裁剪，每次会裁剪的比例为 [ch_sparsity / iterative_steps]
        macs, nparams = tp.utils.count_ops_and_params(myModel_BaseBEVBackbone, example_inputs)
        print("  Iter %d/%d, Params: %.2f M => %.2f M" % (i+1, iterative_steps, base_nparams / 1e6, nparams / 1e6))
        print("  Iter %d/%d, MACs: %.2f G => %.2f G"% (i+1, iterative_steps, base_macs / 1e9, macs / 1e9))
        print("  Iter %d/%d, Pruning ratio: %.2f " % (i+1, iterative_steps, nparams / base_nparams))
        
    # 再次查看模型结构以及参数量    
    # print(myModel_BaseBEVBackbone)
    # summary(myModel_BaseBEVBackbone, input_data=example_inputs)
    
    # 替换原先模型，并Finetuning（重训练）
    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=demo_dataset)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=False)
    
    # 评估剪枝前模型性能
    model = model.cuda().eval()
    model_no_pruner = copy.deepcopy(model)
    # print(model) # 剪枝前的模型结构
    # print(model.module_list[2])
    
    # 方式一
    # model.module_list[2] = copy.deepcopy(myModel_BaseBEVBackbone)  # TBD 需对齐子模块输入输出
    
    # 方式二: 直接替换内部modules
    model.backbone_2d.blocks = copy.deepcopy(myModel_BaseBEVBackbone.blocks)
    model.backbone_2d.deblocks = copy.deepcopy(myModel_BaseBEVBackbone.deblocks)
    # print(model.backbone_2d) # 剪枝后的模型结构
    # print(model)
    
    
    # 评估剪枝后模型性能
    # model.cuda().eval()
    # eval_model(model, cfg, args, logger)
    # # —————————————— 剪枝流程结束 —————————————— # 
    
    model = model.cuda().eval()
    model_after_pruner = copy.deepcopy(model)
    
    # 比较剪枝前后模型运算效率
    
    
    logger.info(f'Display Model Before Pruning : \n {model_no_pruner}')
    t1 = 0
    count = 0
    with torch.no_grad():
        for idx, data_dict in enumerate(demo_dataset):
            data_dict = demo_dataset.collate_batch([data_dict])
            load_data_to_gpu(data_dict)
            
            pred_dicts, recall_dicts, time_info = model_no_pruner.forward(data_dict)
            
            if idx > 10:
                count = count + 1
                for key, value in time_info.items():
                    t1[key] = t1[key] + value
            else:
                t1 = time_info

            if count >= 100:
                break
    
    logger.info(f'Display Model After Pruning : \n {model_after_pruner}')        
    t2 = 0    
    count = 0
    with torch.no_grad():
        for idx, data_dict in enumerate(demo_dataset):
            data_dict = demo_dataset.collate_batch([data_dict])
            load_data_to_gpu(data_dict)
        
            pred_dicts, recall_dicts, time_info = model_after_pruner.forward(data_dict)
            
            if idx > 10:
                count = count + 1
                for key, value in time_info.items():
                    t2[key] = t2[key] + value
            else:
                t2 = time_info

            if count >= 100:
                break
    
    
    for key, val in t1.items():
        # print(key, val)
        print("SubModule: %-*s T1: %-*.4f  T1: %-*.4f fact: %-*.4f" % (30, key, 10, val, 10, t2[key], 10, val / t2[key]))
    
    # 剪枝后评估模型效果
    model = model.cuda().eval()
    # print(next(model.parameters()).device) 
    eval_model(model, cfg, args, logger)
        
    # # 重新训练
    re_trainModel(model)
    
    # # 重新训练后再次评估模型效果
    eval_model(model, cfg, args, logger)


    