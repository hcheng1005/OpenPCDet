import numpy as np
import os
import open3d

# vis = open3d.visualization.Visualizer()
# vis.create_window()
 
base_path = r'/media/charles/ShareDisk/00myDataSet/KITTI/2011_09_26_drive_0056_extract/2011_09_26/2011_09_26_drive_0056_extract/velodyne_points/data'
files = os.listdir(base_path)
files.sort(key=lambda x: int(x.split('.')[0]))

for file in files:
    #获取文件所属目录
    # print(root)
    #获取文件路径
    points = np.loadtxt(os.path.join(base_path,file), dtype=np.float32)
    points = points.reshape(-1, 4)
        
    points[:, 3] = 0 
    np.save(os.path.join(base_path, os.path.splitext(file)[0]), points) 
            