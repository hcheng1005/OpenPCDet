# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import onnx
import numpy as np
import onnx_graphsurgeon as gs

@gs.Graph.register()
def replace_with_clip(self, inputs, outputs):
    for inp in inputs:
        inp.outputs.clear()

    for out in outputs:
        out.inputs.clear()

    op_attrs = dict()
    op_attrs["dense_shape"] = np.array([496,432])

    return self.layer(name="PPScatter_0", op="PPScatterPlugin", inputs=inputs, outputs=outputs, attrs=op_attrs)

def loop_node(graph, current_node, loop_time=0):
  for i in range(loop_time):
    next_node = [node for node in graph.nodes if len(node.inputs) != 0 and len(current_node.outputs) != 0 and node.inputs[0] == current_node.outputs[0]][0]
    current_node = next_node
  return next_node

def simplify_postprocess(onnx_model):
  print("Use onnx_graphsurgeon to adjust postprocessing part in the onnx...")
  graph = gs.import_onnx(onnx_model)

  output_01 = gs.Variable(name="11", dtype=np.float32)
  output_02 = gs.Variable(name="12", dtype=np.float32)
  output_03 = gs.Variable(name="13", dtype=np.float32)
  output_04 = gs.Variable(name="14", dtype=np.float32)
  output_05 = gs.Variable(name="15", dtype=np.float32)
  output_06 = gs.Variable(name="16", dtype=np.float32)
  output_011 = gs.Variable(name="21", dtype=np.float32)
  output_021 = gs.Variable(name="22", dtype=np.float32)
  output_031 = gs.Variable(name="23", dtype=np.float32)
  output_041 = gs.Variable(name="24", dtype=np.float32)
  output_051 = gs.Variable(name="25", dtype=np.float32)
  output_061 = gs.Variable(name="26", dtype=np.float32)

  tmap = graph.tensors()
  new_inputs = [tmap["voxels"], tmap["voxel_idxs"], tmap["voxel_num"]]
  new_outputs = [output_01, output_02, output_03,output_04, output_05, output_06,
                 output_011, output_021, output_031,output_041, output_051, output_061]

  for inp in graph.inputs:
    if inp not in new_inputs:
      inp.outputs.clear()

  for out in graph.outputs:
    out.inputs.clear()

  #  找到链接ConvTranspose的node
  first_ConvTranspose_node = [node for node in graph.nodes if node.op == "ConvTranspose"][0] 
  
  # print('first_ConvTranspose_node', first_ConvTranspose_node)
  # print('first_ConvTranspose_node', first_ConvTranspose_node.outputs[0])
  
  current_node = first_ConvTranspose_node
  for i in range(3):
    # print('__________________________________________________________________')
    # print(current_node)
    for node in graph.nodes: 
      if len(node.inputs) != 0 and len(current_node.outputs) != 0 :
        if node.op == "Concat":
            if node.inputs[1] == current_node.outputs[0]:
              # print([node][0])
              next_node = [node][0]
        else:
            if node.inputs[0] == current_node.outputs[0]:
              # print([node][0])
              next_node = [node][0]

    current_node = next_node
    # print(current_node.outputs[0])
    
  concat_node = current_node
  assert concat_node.op == "Concat"
  # concat_node = loop_node(graph, first_ConvTranspose_node, 3)
  # print(concat_node)
  # assert concat_node.op == "Concat"

  conv_node_after_concat = [node for node in graph.nodes if len(node.inputs) != 0 and len(concat_node.outputs) != 0 and node.inputs[0] == concat_node.outputs[0]][0]
  # print('conv_node_after_concat \n', conv_node_after_concat)
  
  # print(conv_node_after_concat.outputs)
  
  first_node_after_concat = [node for node in graph.nodes if len(node.inputs) != 0 and len(conv_node_after_concat.outputs) != 0 and node.inputs[0] == conv_node_after_concat.outputs[0]][0]
  # print('first_node_after_concat \n', first_node_after_concat)
  # print(len(first_node_after_concat))
  
  first_node_after_relu = [node for node in graph.nodes if len(node.inputs) != 0 and len(first_node_after_concat.outputs) != 0 and node.inputs[0] == first_node_after_concat.outputs[0]]
  # print('first_node_after_concat \n', first_node_after_relu)
  # print(len(first_node_after_relu))

  for i in range(len(first_node_after_relu)):
    transpose_node = loop_node(graph, first_node_after_relu[i], 2)
    print('transpose_node \n', transpose_node)
    assert transpose_node.op == "Conv"
    transpose_node.outputs = [new_outputs[i]] # 重新设定模型输出节点与位置

  graph.inputs = new_inputs
  graph.outputs = new_outputs
  graph.cleanup().toposort()
  
  return gs.export_onnx(graph)


def simplify_preprocess(onnx_model):
  print("Use onnx_graphsurgeon to modify onnx...")
  graph = gs.import_onnx(onnx_model)

  tmap = graph.tensors()
  print(tmap)
  MAX_VOXELS = tmap["voxels"].shape[0]

  # voxels: [V, P, C']
  # V is the maximum number of voxels per frame
  # P is the maximum number of points per voxel
  # C' is the number of channels(features) per point in voxels.
  input_new = gs.Variable(name="voxels", dtype=np.float32, shape=(MAX_VOXELS, 32, 10))

  # voxel_idxs: [V, 4]
  # V is the maximum number of voxels per frame
  # 4 is just the length of indexs encoded as (frame_id, z, y, x).
  X = gs.Variable(name="voxel_idxs", dtype=np.int32, shape=(MAX_VOXELS, 4))

  # voxel_num: [1]
  # Gives valid voxels number for each frame
  Y = gs.Variable(name="voxel_num", dtype=np.int32, shape=(1,))

  first_node_after_pillarscatter = [node for node in graph.nodes if node.op == "Conv"][0]

  first_node_pillarvfe = [node for node in graph.nodes if node.op == "MatMul"][0]

  next_node = current_node = first_node_pillarvfe
  for i in range(6):
    next_node = [node for node in graph.nodes if node.inputs[0] == current_node.outputs[0]][0]
    if i == 5:              # ReduceMax
      current_node.attrs['keepdims'] = [0]
      break
    current_node = next_node

  last_node_pillarvfe = current_node

  #merge some layers into one layer between inputs and outputs as below
  graph.inputs.append(Y)
  inputs = [last_node_pillarvfe.outputs[0], X, Y]
  outputs = [first_node_after_pillarscatter.inputs[0]]
  graph.replace_with_clip(inputs, outputs)

  # Remove the now-dangling subgraph.
  graph.cleanup().toposort()

  #just keep some layers between inputs and outputs as below
  graph.inputs = [first_node_pillarvfe.inputs[0] , X, Y]
  graph.outputs = [tmap["11"], tmap["12"], tmap["13"],tmap["14"], tmap["15"], tmap["16"],
                   tmap["21"], tmap["22"], tmap["23"],tmap["24"], tmap["25"], tmap["26"]]

  graph.cleanup()

  #Rename the first tensor for the first layer 
  graph.inputs = [input_new, X, Y]
  first_add = [node for node in graph.nodes if node.op == "MatMul"][0]
  first_add.inputs[0] = input_new

  graph.cleanup().toposort()

  return gs.export_onnx(graph)

if __name__ == '__main__':
    mode_file = "pointpillar-native-sim.onnx"
    simplify_preprocess(onnx.load(mode_file))
