"""
load weight of Video-Nonlocal-Net style checkpoint (caffe2)
into mmaction2 style checkpoint (PyTorch)

input:  caffe2 checkpoint (load from)
        mmaction2 checkpoint (to be overwritten)
output: mmaction2 checkpoint (result)
"""

# conv1_w                                 (64, 3, 1, 7, 7)
# → backbone.conv1.conv.weight            torch.Size([64, 3, 7, 7])
# res_conv1_bn_s                          (64,)
# → backbone.conv1.bn.weight              torch.Size([64])
# res_conv1_bn_b                          (64,)
# → backbone.conv1.bn.bias                torch.Size([64])
# res_conv1_bn_rm                         (64,)
# → backbone.conv1.bn.running_mean        torch.Size([64])
# res_conv1_bn_riv                        (64,)
# → backbone.conv1.bn.running_var         torch.Size([64])

# res{$y+1}_{$x}_branch2{a,b,c}_w                       (64, 64, 1, 1, 1)
# → backbone.layer{$y}.{$x}.conv{1,2,3}.conv.weight     torch.Size([64,64,1,1])
# res{$y+1}_{$x}_branch2{a,b,c}_bn_s                    (64,)
# → backbone.layer{$y}.{$x}.conv{1,2,3}.bn.weight       torch.Size([64])
# res{$y+1}_{$x}_branch2{a,b,c}_bn_b                    (64,)
# → backbone.layer{$y}.{$x}.conv{1,2,3}.bn.bias         torch.Size([64])
# res{$y+1}_{$x}_branch2{a,b,c}_bn_rm                   (64,)
# → backbone.layer{$y}.{$x}.conv{1,2,3}.bn.running_mean torch.Size([64])
# res{$y+1}_{$x}_branch2{a,b,c}_bn_riv                  (64,)
# → backbone.layer{$y}.{$x}.conv{1,2,3}.bn.running_var  torch.Size([64])

# res{$y+1}_{$x}_branch1_w                              (256, 64, 1, 1, 1)
# → backbone.layer{$y}.{$x}.downsample.conv.weight     torch.Size([256,64,1,1])
# res{$y+1}_{$x}_branch1_bn_s                           (256,)
# → backbone.layer{$y}.{$x}.downsample.bn.weight        torch.Size([256])
# res{$y+1}_{$x}_branch1_bn_b                           (256,)
# → backbone.layer{$y}.{$x}.downsample.bn.bias          torch.Size([256])
# res{$y+1}_{$x}_branch1_bn_rm                          (256,)
# → backbone.layer{$y}.{$x}.downsample.bn.running_mean  torch.Size([256])
# res{$y+1}_{$x}_branch1_bn_riv                         (256,)
# → backbone.layer{$y}.{$x}.downsample.bn.running_var   torch.Size([256])

# pred_w                                                (400, 2048)
# → cls_head.fc_cls.weight                              torch.Size([400, 2048])
# pred_b                                                (400,)
# → cls_head.fc_cls.bias                                torch.Size([400])

# build a dict, convert left string to right string
# for example:
# 'res4_1_branch2a_w' -> 'backbone.layer3.1.conv1.conv.weight'
# 'res4_1_branch2b_w' -> 'backbone.layer3.1.conv2.conv.weight'

import pickle
import re

import torch


def convert_name(caffe2_name):
    if caffe2_name == 'conv1_w':
        return 'backbone.conv1.conv.weight'
    elif caffe2_name == 'res_conv1_bn_s':
        return 'backbone.conv1.bn.weight'
    elif caffe2_name == 'res_conv1_bn_b':
        return 'backbone.conv1.bn.bias'
    elif caffe2_name == 'res_conv1_bn_rm':
        return 'backbone.conv1.bn.running_mean'
    elif caffe2_name == 'res_conv1_bn_riv':
        return 'backbone.conv1.bn.running_var'
    elif caffe2_name == 'pred_w':
        return 'cls_head.fc_cls.weight'
    elif caffe2_name == 'pred_b':
        return 'cls_head.fc_cls.bias'
    else:
        caffe2_name = re.sub(r"^res([0-9])_([0-9])_branch2([a-z])_",
                             r"backbone.layer\1.\2.conv\3.", caffe2_name)
        caffe2_name = re.sub(r"^res([0-9])_([0-9])_branch1_",
                             r"backbone.layer\1.\2.downsample.", caffe2_name)
        caffe2_name = re.sub("conva", "conv1", caffe2_name)
        caffe2_name = re.sub("convb", "conv2", caffe2_name)
        caffe2_name = re.sub("convc", "conv3", caffe2_name)
        caffe2_name = re.sub(r"w$", "conv.weight", caffe2_name)
        caffe2_name = re.sub(r"bn_s$", "bn.weight", caffe2_name)
        caffe2_name = re.sub(r"bn_b$", "bn.bias", caffe2_name)
        caffe2_name = re.sub(r"bn_rm$", "bn.running_mean", caffe2_name)
        caffe2_name = re.sub(r"bn_riv$", "bn.running_var", caffe2_name)
        caffe2_name = re.sub(r"(?<=layer)\d+",
                             lambda x: str(int(x.group(0)) - 1), caffe2_name)
        return caffe2_name


pthmodel = torch.load('mmaction2_ckpt_input.pth')

with open('video_nonlocal_ckpt.pkl', "rb") as file:
    caffe2model = pickle.load(file, encoding='latin1')
c2keylst = list(caffe2model['blobs'].keys())

for k in c2keylst:
    if (k not in {'model_iter', 'lr'}) and ('momentum' not in k):
        pthname = convert_name(k)
        weight = torch.from_numpy(caffe2model['blobs'][k]).reshape_as(
            pthmodel['state_dict'][pthname])
        pthmodel['state_dict'][pthname] = weight

torch.save(pthmodel, 'mmaction2_ckpt_output.pth')
