import numpy as np
from PIL import Image
import cv2
import torch
import torch_dct as dct
def rgb_to_dct(image_path, output_path):
    # 读取RGB图像
    image = Image.open(image_path)
    
    rgb_array = np.array(image)
    rgb_array = np.float32(rgb_array)
    # rgb_array = (rgb_array-127.5)/127.5
    print(np.max(rgb_array))
    # 对每个颜色通道进行DCT变换
    dct_array = np.zeros(rgb_array.shape, dtype=np.float32)
    idct_array = np.zeros(rgb_array.shape, dtype=np.float32)
    for i in range(3):  # 遍历RGB通道
        dct_array[:, :, i] = np.float32(cv2.dct(rgb_array[:, :, i]))
    for i in range(3):  # 遍历RGB通道
        idct_array[:, :, i] = np.float32(cv2.idct(dct_array[:, :, i]))
    print(np.min(dct_array))
    # 保存DCT结果为图像文件
    dct_image = Image.fromarray(np.uint8(dct_array))
    dct_image.save(output_path)
    idct_image = Image.fromarray(np.uint8(idct_array))
    idct_image.save(output_path)

    img = torch.from_numpy(rgb_array).float().to("cuda").unsqueeze(0)
    img= img.permute(0, 3, 1, 2)
    
    dct_img_tensor0 = torch.from_numpy(dct_array).float().to("cuda").unsqueeze(0)
    dct_img_tensor0= dct_img_tensor0.permute(0, 3, 1, 2)
    dct_img_tensor1 = torch.zeros_like(dct_img_tensor0)
    img_tensor1 = torch.zeros_like(dct_img_tensor0)
    for i in range(3):
        dct_img_tensor1[:,i:,:,:] = dct.dct_2d(img[:,i:,:,:] , norm='ortho')
    for i in range(3):
        img_tensor1[:,i:,:,:]= dct.idct_2d(dct_img_tensor1[:,i:,:,:], norm='ortho')
    print(torch.min(dct_img_tensor0-dct_img_tensor1))
    print(torch.min(img-img_tensor1))
    print(torch.max(dct_img_tensor0),torch.max(img))
# 示例用法
input_image_path = "./demo.png"
output_image_path = "output_image.jpg"
rgb_to_dct(input_image_path, output_image_path)