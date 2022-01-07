from model_wrapper import NB101Wrapper
from pynbs.nasbench101.model import Network
from pynbs.nasbench101.model_spec import *
import configs
import torch
import numpy as np
import torch.nn as nn
import onnx

arch = [[[0, 1, 0, 0, 0, 0, 0],
         [0, 0, 1, 0, 0, 1, 0],
         [0, 0, 0, 1, 1, 1, 0],
         [0, 0, 0, 0, 0, 1, 0],
         [0, 0, 0, 0, 0, 1, 0],
         [0, 0, 0, 0, 0, 0, 1],
         [0, 0, 0, 0, 0, 0, 0]],
        ['input',
         'conv1x1-bn-relu',
         'conv1x1-bn-relu',
         'conv3x3-bn-relu',
         'conv1x1-bn-relu',
         'maxpool3x3',
         'output']]

learning_rate = 1e-1
search_space = 'nasbench101'
momentum = 0.9
weight_decay = 4e-5
total_iters = 100
eval_interval = 100
init_w_type = 'none'
init_b_type = 'none'
dataset = 'cifar10'
batch_size = 16
output_size = 8
# config = 'CONF_NB101'
config = configs.CONF_NB101
nds_path = '../../GenNAS/data/nds_data/'
device = torch.device("cuda:" + "0" if torch.cuda.is_available() else "cpu")

init_channels = 16
last_channels = 64
last_channels = np.asarray([config['last_channel_l0'] * last_channels, config['last_channel_l1'] * last_channels,
                            config['last_channel_l2'] * last_channels]).astype(int)
last_channels = last_channels.tolist()

# myWrapper = NB101Wrapper(arch, init_channels, last_channels)
# # myWrapper_script = torch.jit.script(myWrapper)



matrix = arch[0]
matrix = torch.tensor(matrix)
ops = arch[1]
output_size = 8
num_stacks = 3
num_modules_per_stack = 3
num_labels = 10
stem_out_channels = init_channels
out_channels = stem_out_channels

spec = ModelSpec(matrix, ops, data_format='channels_last')
myModel_wrapper = Network(spec, stem_out_channels, num_stacks, num_modules_per_stack, num_labels)
myModel_wrapper.eval()


myModel_wrapper_scripted = torch.jit.script(myModel_wrapper)
# myModel_wrapper_scripted.save('myModel_wrapper_scripted.pth')
# x = torch.randn(batch_size, 3, 32, 32, requires_grad=True)
# # Export the model
# torch.onnx.export(myModel_wrapper,               # model being run
#                   x,                         # model input (or a tuple for multiple inputs)
#                   "myGoodModel.onnx",   # where to save the model (can be a file or file-like object)
#                   export_params=True,        # store the trained parameter weights inside the model file
#                   opset_version=10,          # the ONNX version to export the model to
#                   do_constant_folding=True,  # whether to execute constant folding for optimization
#                   input_names = ['input'],   # the model's input names
#                   output_names = ['output'], # the model's output names
#                   dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
#                                 'output' : {0 : 'batch_size'}})
#
#
# onnx_model = onnx.load("myGoodModel.onnx")
# onnx.checker.check_model(onnx_model)
# torch.jit.script()