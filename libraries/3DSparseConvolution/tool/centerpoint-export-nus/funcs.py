import torch
import collections
from pcdet.models.backbones_3d import VoxelResBackBone8x

import spconv
from spconv.pytorch import SparseSequential
from spconv.pytorch import conv
from spconv.pytorch.conv import SubMConv3d, SparseConv3d

from spconv.pytorch.conv import SparseConvolution, SparseConvTensor

# from det3d.models.backbones.scn import SparseBasicBlock
from pcdet.models.backbones_3d import SparseBasicBlock
import cumm.tensorview as tv
import numpy as np

def make_new_repr(old_repr):
    def new_repr(self):
        s = old_repr(self)
        if self.act_type is not None:
            p = s.rfind(")")
            s = s[:p] + f', act={self.act_type}' + s[p:]
        return s
    return new_repr

# setup repr function, add activation
conv.SparseConvolution.__repr__ = make_new_repr(conv.SparseConvolution.__repr__)

def load_scn_backbone_checkpoint(model, file):

    device   = next(model.parameters()).device    
    ckpt     = torch.load(file, map_location=device)["model_state"]
    new_ckpt = collections.OrderedDict()
    for key, val in ckpt.items():
        if key.startswith("backbone_3d."):
            newkey = key[key.find(".")+1:]
            
            if (len(val.shape) == 5):
                # N,W,X,Y,Z = val.shape
                # val = val.view(Z,N,W,X,Y)
                val = val.permute(4, 0, 1, 2, 3)
            new_ckpt[newkey] = val

    model.load_state_dict(new_ckpt)
    return model

def fuse_bn_weights(conv_w_OKI, conv_b, bn_rm, bn_rv, bn_eps, bn_w, bn_b):
    NDim = conv_w_OKI.ndim - 2
    permute = [0, NDim+1] + [i+1 for i in range(NDim)]
    conv_w_OIK = conv_w_OKI.permute(*permute)
    # OIDHW
    if conv_b is None:
        conv_b = torch.zeros_like(bn_rm)
    if bn_w is None:
        bn_w = torch.ones_like(bn_rm)
    if bn_b is None:
        bn_b = torch.zeros_like(bn_rm)
    bn_var_rsqrt = torch.rsqrt(bn_rv + bn_eps)

    conv_w_OIK = conv_w_OIK * (bn_w * bn_var_rsqrt).reshape([-1] + [1] * (len(conv_w_OIK.shape) - 1))
    conv_b = (conv_b - bn_rm) * bn_var_rsqrt * bn_w + bn_b
    permute = [0,] + [i+2 for i in range(NDim)] + [1,]
    conv_w_OKI = conv_w_OIK.permute(*permute).contiguous()
    return torch.nn.Parameter(conv_w_OKI), torch.nn.Parameter(conv_b)

def fuse_bn(conv, bn):
    """
    Given a conv Module `A` and an batch_norm module `B`, returns a conv
    module `C` such that C(x) == B(A(x)) in inference mode.
    """
    #assert(not (conv.training or bn.training)), "Fusion only for eval!"    
    conv.weight, conv.bias = fuse_bn_weights(conv.weight, conv.bias, bn.running_mean, bn.running_var, bn.eps, bn.weight, bn.bias)


def load_checkpoint(model, file, startsname=None):
    device   = next(model.parameters()).device
    ckpt     = torch.load(file, map_location=device)["state_dict"]
    new_ckpt = ckpt
    if startsname is not None:
        new_ckpt = collections.OrderedDict()
        for key, val in ckpt.items():
            if key.startswith(startsname):
                newkey = key[len(startsname)+1:]
                new_ckpt[newkey] = val

    model.load_state_dict(new_ckpt, strict =True)


def new_sparse_basic_block_forward(self):
    def sparse_basic_block_forward(x):
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out.replace_feature(out.features + identity.features)
        out = out.replace_feature(self.relu(out.features))
        return out
    return sparse_basic_block_forward


def fuse_sparse_basic_block(self, is_fuse_bn = False, is_fuse_relu=True):
    self.forward = new_sparse_basic_block_forward(self)
    if is_fuse_relu == True:
        self.conv1.act_type = tv.gemm.Activation.ReLU 

    if is_fuse_bn == True:
        fuse_bn(self.conv1, self.bn1)
        fuse_bn(self.conv2, self.bn2)
        delattr(self, "bn1")
        delattr(self, "bn2")

def layer_fusion_bn(model):

    def set_attr_by_path(m, path, newval):

        def set_attr_by_array(parent, arr):
            if len(arr) == 1: 
                setattr(parent, arr[0], newval)
                return parent

            parent = getattr(parent, arr[0])
            return set_attr_by_array(parent, arr[1:])

        return set_attr_by_array(m, path.split("."))


    for name, module in model.named_modules():
        if isinstance(module, SparseSequential):
            if isinstance(module[0], spconv.conv.SparseConvolution):
                c, b, r = [module[i] for i in range(3)]
                fuse_bn(c, b)

                c = SparseSequential(
                    c,r
                )
                set_attr_by_path(model, name, c)
        elif isinstance(module, SparseBasicBlock):
            fuse_sparse_basic_block(module, is_fuse_bn = True, is_fuse_relu =False)
        elif isinstance(module, torch.nn.ReLU): 
            module.inplace = False
    return model

def layer_fusion(model):
    def set_attr_by_path(m, path, newval):
        def set_attr_by_array(parent, arr):
            if len(arr) == 1: 
                setattr(parent, arr[0], newval)
                return parent
            parent = getattr(parent, arr[0])
            return set_attr_by_array(parent, arr[1:])

        return set_attr_by_array(m, path.split("."))

    for name, module in model.named_modules():
        if isinstance(module, SparseSequential):
            if isinstance(module[0], SubMConv3d) or isinstance(module[0], SparseConv3d):
                c, b, r = [module[i] for i in range(3)]
                fuse_bn(c, b)
                c.act_type = tv.gemm.Activation.ReLU
                set_attr_by_path(model, name, c)
        elif isinstance(module, SparseBasicBlock):
            fuse_sparse_basic_block(module, is_fuse_relu= True, is_fuse_bn= True)
        elif isinstance(module, torch.nn.ReLU): 
            module.inplace = False
    return model