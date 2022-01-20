import json
import sys
import os
import time
import numpy as np
import cv2
import onnx
import onnxruntime
from onnx import numpy_helper

# model_dir = "./mnist"
# model = model_dir + "/model.onnx"
# # path = sys.argv[1]
img_path = "pics/horse.jpg"
onnx_model_path = "cifar_net.onnx"


# Preprocess the image


def predict(img_path, onnx_model_path):
    img = cv2.imread(img_path)
    # img = np.dot(img[..., :3], [0.2023, 0.1994, 0.2010])
    mean = np.array([0.4914, 0.4822, 0.4465])
    std = np.array([0.2023, 0.1994, 0.2010])

    img = cv2.resize(img, dsize=(32, 32), interpolation=cv2.INTER_AREA)
    # img = (img - mean) / std  ## normalization
    img = cv2.normalize(img, None, -1, 1, norm_type=cv2.NORM_MINMAX)

    img.resize((1, 3, 32, 32))

    data = json.dumps({'data': img.tolist()})
    data = np.array(json.loads(data)['data']).astype('float32')
    session = onnxruntime.InferenceSession(onnx_model_path)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    # print(input_name)
    # print(output_name)

    result = session.run([output_name], {input_name: data})
    prediction = int(np.argmax(np.array(result).squeeze(), axis=0))
    classes = ['bird', 'airplane', 'automobile', 'horse', 'cat', 'deer', 'frog', 'ship', 'truck', 'dog']

    print(classes[prediction], result)


predict(img_path, onnx_model_path)
