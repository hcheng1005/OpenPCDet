# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import torch
import onnx
import funcs
import onnx.helper as helper
import numpy as np
import spconv.pytorch as spconv
import cumm.tensorview as tv
from spconv.core import ConvAlgo
import torch.nn as nn

avoid_reuse_container = []
obj_to_tensor_id = {}
nodes = []
initializers = []
enable_trace = False

def register_node(fn):

    fnnames   = fn.split(".")
    fn_module = eval(".".join(fnnames[:-1]))
    fn_name   = fnnames[-1]
    oldfn = getattr(fn_module, fn_name)
    
    def make_hook(bind_fn):

        ilayer = 0
        def internal_forward(self, *args):
            global enable_trace

            if not enable_trace:
                return oldfn(self, *args)

            global avoid_reuse_container
            nonlocal ilayer

            # Use the enable_trace flag to avoid internal trace calls
            enable_trace = False
            y = oldfn(self, *args)
            bind_fn(self, ilayer, y, *args)
            enable_trace = True

            avoid_reuse_container.extend(list(args) + [y]) 
            ilayer += 1
            return y

        setattr(fn_module, fn_name, internal_forward)
    return make_hook

@register_node("spconv.conv.SparseConvolution.forward")
def symbolic_sparse_convolution(self, ilayer, y, x):
    register_tensor(y)
    print(f"   --> SparseConvolution{ilayer}[{'subm' if self.subm else 'conv'}] -> Input {get_tensor_id(x)}, Output {get_tensor_id(y)}")

    if self.transposed:
        output_size = spconv.ops.get_deconv_output_size(
            x.features.size(), self.kernel_size, self.stride, self.padding, self.dilation, self.output_padding
        )
    else:
        output_size = spconv.ops.get_conv_output_size(
            x.features.size(), self.kernel_size, self.stride, self.padding, self.dilation
        )
    
    if self.subm:
        output_size[0] = x.features.size(0)
    
    output_size[1] = self.out_channels
    inputs = [
        get_tensor_id(x), 
        append_initializer(self.weight.data, f"spconv{ilayer}.weight"),
    ]
    
    if self.bias is not None:
        inputs.append(append_initializer(self.bias.data, f"spconv{ilayer}.bias"))
        
    act_type_name = {
        tv.gemm.Activation.ReLU      : "ReLU",
        tv.gemm.Activation.None_     : "None",
        tv.gemm.Activation.Sigmoid   : "Sigmoid",
        tv.gemm.Activation.LeakyReLU : "LeakyReLU"
    }

    output_bound = 200000
    if hasattr(self, "output_bound"):
        output_bound = self.output_bound

    nodes.append(
        helper.make_node(
            "SparseConvolution", inputs, [get_tensor_id(y)], f"conv{ilayer}", 
            ndim = self.ndim,
            input_spatial_shape = x.spatial_shape,
            output_spatial_shape = y.spatial_shape,
            in_channels = self.in_channels,
            out_channels = self.out_channels,
            kernel_size = self.kernel_size,
            output_bound = output_bound,
            stride = self.stride,
            dilation = self.dilation,
            padding = self.padding,
            transposed = self.transposed,
            inverse = False,
            output_padding = self.output_padding,
            groups = self.groups,
            subm = self.subm,
            rulebook = self.indice_key,
            activation = act_type_name[self.act_type],
            input_shape  = x.features.shape,
            output_shape = y.features.shape
        )
    )

@register_node("torch.nn.ReLU.forward")
def symbolic_relu(self, ilayer, y, x):
    register_tensor(y)
    print(f"   --> ReLU{ilayer} -> Input {get_tensor_id(x)}, Output {get_tensor_id(y)}")

    nodes.append(
        helper.make_node(
            "Relu", [get_tensor_id(x)], [get_tensor_id(y)], f"relu{ilayer}"
        )
    )

@register_node("torch.Tensor.__add__")
def symbolic_add(a, ilayer, y, b):
    register_tensor(y)
    print(f"   --> Add{ilayer} -> Input {get_tensor_id(a)} + {get_tensor_id(b)}, Output {get_tensor_id(y)}")

    nodes.append(
        helper.make_node(
            "Add", [get_tensor_id(a), get_tensor_id(b)], [get_tensor_id(y)], f"add{ilayer}"
        )
    )

@register_node("spconv.core.SparseConvTensor.dense")
def node_sparse_conv_tensor_dense(self, ilayer, y):
    register_tensor(y)
    print(f"   --> ToDense{ilayer}[{self.spatial_shape}][{list(y.size())}] -> Input {get_tensor_id(self)}, Output {get_tensor_id(y)}")
    
    format = "zyx"
    nodes.append(
        helper.make_node(
            "ScatterDense", [get_tensor_id(self)], [get_tensor_id(y)], f"scatter{ilayer}",
            input_spatial_shape = self.spatial_shape,
            format              = format,
            output_shape        = list(y.size())
        )
    )

@register_node("torch.Tensor.reshape")
def node_view(self, ilayer, y, *dims):
    register_tensor(y)
    print(f"   --> Reshape{ilayer}[{dims}] -> Input {get_tensor_id(self)}, Output {get_tensor_id(y)}")

    nodes.append(
        helper.make_node(
            "Reshape", [get_tensor_id(self)], [get_tensor_id(y)], f"reshape{ilayer}",
            dims = dims
        )
    )

@register_node("torch.Tensor.permute")
def node_permute(self, ilayer, y, *dims):
    register_tensor(y)
    print(f"   --> Permute{ilayer}[{dims}][{list(y.shape)}] -> Input {get_tensor_id(self)}, Output {get_tensor_id(y)}")

    nodes.append(
        helper.make_node(
            "Transpose", [get_tensor_id(self)], [get_tensor_id(y)], f"transpose{ilayer}",
            dims = dims
        )
    )

def append_initializer(value, name):
    initializers.append(
        helper.make_tensor(
            name=name,
            data_type=helper.TensorProto.DataType.FLOAT16,
            dims=list(value.shape),
            vals=value.cpu().data.numpy().astype(np.float16).tobytes(),
            raw=True
        )
    )
    return name

def __obj_to_id(obj):
    idd = id(obj)
    if isinstance(obj, spconv.SparseConvTensor):
        idd = id(obj.features)
    return idd

def set_obj_idd_assame(a_already_has_idd, b_no_idd):
    global obj_to_tensor_id
    aidd = __obj_to_id(a_already_has_idd)
    bidd = __obj_to_id(b_no_idd)
    
    assert aidd in obj_to_tensor_id, "A is not in tensor map"
    assert bidd not in obj_to_tensor_id, "B is already in tensor map"
    obj_to_tensor_id[bidd] = obj_to_tensor_id[aidd]

def register_tensor(obj):
    global obj_to_tensor_id
    obj_to_tensor_id[__obj_to_id(obj)] = str(len(obj_to_tensor_id))
    
def get_tensor_id(obj):
    idd = __obj_to_id(obj)
    assert idd in obj_to_tensor_id, "ops!!!😮 Cannot find the tensorid of this object. this means that some operators are not being traced. You need to confirm it."
    return obj_to_tensor_id[idd]

def make_model_forward_hook(self, inverse_indices=False):
    def impl(voxel_features, coors, batch_size, **kwargs):
        coors = coors.int()
        input_sp_tensor = spconv.SparseConvTensor(
            voxel_features, coors, self.sparse_shape, batch_size
        )
        
        x = self.conv_input(input_sp_tensor)
        
        x_conv1 = self.conv1(x)
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        x_conv4 = self.conv4(x_conv3)

        out = self.conv_out(x_conv4)
        spatial_features = out.dense()
        
        N, C, Z, Y, X = spatial_features.shape
        # spatial_features = spatial_features.permute(0, 1, 2, 4, 3)
        spatial_features = spatial_features.reshape(N, C * Z, Y, X)
        
        return spatial_features
    
    return impl
            
            
def export_onnx(model, voxels, coors, batch_size, spatial_shape, save_onnx, save_tensor):

    global avoid_reuse_container, tensor_map, nodes, initializers, enable_trace
    avoid_reuse_container = []
    tensor_map = {}
    nodes = []
    initializers = []
        
    model.forward = make_model_forward_hook(model, True)

    print("Tracing model inference...")
    print("> Do inference...")
    with torch.no_grad():
        register_tensor(voxels)
        
        batch_dict = {}
        batch_dict['voxel_features'] = voxels
        batch_dict['voxel_coords'] = coors
        batch_dict['batch_size'] = batch_size
         
        enable_trace = True
        y = model(voxels, coors, batch_size)

        enable_trace = False
        
        # print(sum(sum(sum(y))))

    if save_tensor is not None:
        print("> Do save tensor, The purpose of this operation is to verify the inference result of C++")
        print(f"   --> Save inference input voxels to {save_tensor}.voxels, voxels.shape = {voxels.shape}")
        funcs.save_tensor(voxels, f"{save_tensor}.voxels")

        print(f"   --> Save inference input coors to {save_tensor}.coors, coors.shape = {coors.shape}")
        funcs.save_tensor(coors,  f"{save_tensor}.coors")

        print(f"   --> Save inference output to {save_tensor}.output, output.shape = {y.shape}")
        funcs.save_tensor(y,      f"{save_tensor}.output")
        
        print(f"   --> Save spatial_shape is {spatial_shape}, batch size is {batch_size}")
        print(f"   --> Save spatial_shape and batch size to {save_tensor}.info")
        funcs.save_tensor([batch_size] + spatial_shape,      f"{save_tensor}.info")

    print("Tracing done!")

    inputs = [
        helper.make_value_info(
            name="0",
            type_proto=helper.make_tensor_type_proto(
                elem_type=helper.TensorProto.DataType.FLOAT16,
                shape=voxels.size()
            )
        )
    ]

    outputs = [
        helper.make_value_info(
            name=get_tensor_id(y),
            type_proto=helper.make_tensor_type_proto(
                elem_type=helper.TensorProto.DataType.FLOAT16,
                shape=y.size()
            )
        )
    ]

    graph = helper.make_graph(
        name="scn",
        inputs=inputs,
        outputs=outputs,
        nodes=nodes,
        initializer=initializers
    )

    opset = [
        helper.make_operatorsetid("ai.onnx", 11)
    ]

    model = helper.make_model(graph, opset_imports=opset, producer_name="pytorch", producer_version="1.9")
    
    # print(model)
    onnx.save_model(model, save_onnx)
    print(f"🚀 The export is completed. ONNX save as {save_onnx} 🤗, Have a nice day~")

    # clean memory
    avoid_reuse_container = []
    tensor_map = {}
    nodes = []
    initializers = []