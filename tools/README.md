# 使用总结

## 训练KITTI数据集
```
OpenPCDet
├── data
│   ├── kitti
│   │   │── ImageSets
│   │   │── training
│   │   │   ├──calib & velodyne & label_2 & image_2 & (optional: planes) & (optional: depth_2)
│   │   │── testing
│   │   │   ├──calib & velodyne & image_2
├── pcdet
├── tools
```

### 1.数据集制作
```python 
python -m pcdet.datasets.kitti.kitti_dataset create_kitti_infos tools/cfgs/dataset_configs/kitti_dataset.yaml
```
### 2. 数据集软连接
```cmd
cd openpcddet
ln -s DATA_SOURCE_PATH ./data/kitti/training
ln -s DATA_SOURCE_PATH ./data/kitti/testing
```
### 3. 运行train
```cmd
cd tools/
```
```python
python train.py --cfg_file="./cfgs/kitti_models/pointpillar.yaml"
```

### 4. 评估
TBD