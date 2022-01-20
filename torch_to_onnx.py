from PIL import Image
from matplotlib import pyplot as plt
import torch
import torchvision.transforms as transforms

net = torch.load("cifar_net.pth")

img_path = "pics/horse.jpg"
image = Image.open(img_path).resize((32, 32))

mean = [0.4914, 0.4822, 0.4465]
std = [0.2023, 0.1994, 0.2010]

size, pad = 32, 4
resize = size
test_transform = transforms.Compose([
    transforms.Resize(resize),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])

input_tensor = test_transform(image)
input_batch = input_tensor.unsqueeze(0)  # create a mini-batch as expected by the model [1,3,32,32]

# plt.imshow(image)
# plt.show()

# Export the model
torch.onnx.export(net,  # model being run
                  input_batch,  # model input (or a tuple for multiple inputs)
                  "cifar_net.onnx",  # where to save the model (can be a file or file-like object)
                  export_params=True,  # store the trained parameter weights inside the model file
                  opset_version=10,  # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names=['input'],  # the model's input names
                  output_names=['output'],  # the model's output names
                  dynamic_axes={'input': {0: 'batch_size'},  # variable length axes
                                'output': {0: 'batch_size'}})
