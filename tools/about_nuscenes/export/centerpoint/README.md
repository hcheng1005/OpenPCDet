# Export CenterPoint

## 参考工程

https://github.com/NVIDIA-AI-IOT/Lidar_AI_Solution/tree/master/CUDA-CenterPoint

Step1:
cd path/openpcdet/tool

```bash
python about_nuscenes/export/centerpoint/export_neck_head.py

python about_nuscenes/export/centerpoint/export_scn.py
```

Step2:
cd path/openpcdet/tool/about_nuscenes/export/centerpoint

```bash
bash build_trt.sh 
```

## ROS部署
TBD