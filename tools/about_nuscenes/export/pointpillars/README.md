# Export PointPillars(MultiHead)

参考： https://github.com/hcheng1005/PointPillars_MultiHead_40FPS

# 导出ONNX
Done

# ONNX2TRT
```bash
onnx2trt cbgs_pp_multihead_pfe.onnx -o cbgs_pp_multihead_pfe.trt -b 1 -d 16 
onnx2trt cbgs_pp_multihead_backbone.onnx -o cbgs_pp_multihead_backbone.trt -b 1 -d 16 
```