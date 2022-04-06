
from pathlib import Path
import json
import os
from sqlite3 import Timestamp

import  numpy as np
import  cv2
from sympy import false, true

import time
import open3d


# vis = open3d.visualization.Visualizer()
# vis.create_window()
# vis.get_render_option().point_size = 1.0
# vis.get_render_option().background_color = np.zeros(3)

#数据地址
data_root_path = '/home/charles/myDataSet/nuScenes/v1.0-mini/'
data_type = 'v1.0-mini'

sample_info_path = data_root_path + data_type + '/' + 'sample.json'
sample_data_info_path = data_root_path + data_type + '/' + 'sample_data.json'
ego_info_path = data_root_path + data_type + '/' + 'ego_pose.json'

with open(sample_info_path,'r') as load_f:
    sample_dict = json.load(load_f)

with open(sample_data_info_path,'r') as load_f:
    sample_data_dict = json.load(load_f)

with open(ego_info_path,'r') as load_f:
    ego_info_dict = json.load(load_f)

# 遍历sample JSON文件并显示点云信息
# for j in range(sample_dict.__len__()):
#     sample_token = sample_dict[j]['token']
#     for j2 in range(sample_data_dict.__len__()):
#         if sample_data_dict[j2]['sample_token'] == sample_token:
#             if '__LIDAR_TOP__' in sample_data_dict[j2]['filename']:
#                 token = sample_dict[j]['next'] #获取下一帧
#                 timestamp = sample_dict[j]['timestamp']
#                 LiDARFile = sample_data_dict[j2]['filename']
#                 ego_pose_token = sample_data_dict[j2]['ego_pose_token']
#                 break
    
#     # 数据处理

#     points = np.fromfile(data_root_path + '/' + LiDARFile, dtype=np.float32, count=-1).reshape([-1, 5])[:, :4]
#     pts = open3d.geometry.PointCloud()
#     pts.points = open3d.utility.Vector3dVector(points[:, :3])

#     vis.add_geometry(pts)
#     vis.poll_events()
#     vis.update_renderer()
#     time.sleep(0.05)
#     vis.clear_geometries()

#遍历sweep下的文件，寻找对应的摄像头、ego、时间戳等信息
base_path = data_root_path + 'sweeps/LIDAR_TOP/'
files = os.listdir(base_path)
# files.sort(key=lambda x: int(x.split('.')[0]))
files.sort(key=lambda x: int(x[42:58])) #根据字符串42-58（转换成数字）进行文件排序
for file in files:
    Timestamp = file[42:58]
    for j in range(sample_data_dict.__len__()):
        if str(sample_data_dict[j]['timestamp']) == Timestamp:
            print(sample_data_dict[j]['ego_pose_token'])
            #进一步获取相关信息
