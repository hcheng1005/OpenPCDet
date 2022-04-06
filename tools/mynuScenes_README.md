## 基于PCDET的3DMOT

### step 1
**检测模型：**

    centerpoint

**检测输出：**
- **CLASS_NAMES**: 
```yaml
    [   'car',
        'truck', 
        'construction_vehicle', 
        'bus', 
        'trailer',
        'barrier', 
        'motorcycle', 
        'bicycle', 
        'pedestrian', 
        'traffic_cone'
        ]
```
- **FEATURES**
```yaml
    [   'x',
        'y', 
        'z', 
        'length', 
        'width',
        'height', 
        ]

    ['class', 
     'score']
```
