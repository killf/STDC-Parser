import numpy as np
import torch
from PIL import Image
import onnxruntime as ort

import os

from solver import Solver
from config import Config

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0],
          [255, 0, 85], [255, 0, 170],
          [0, 255, 0], [85, 255, 0], [170, 255, 0],
          [0, 255, 85], [0, 255, 170],
          [0, 0, 255], [85, 0, 255], [170, 0, 255],
          [0, 85, 255], [0, 170, 255],
          [255, 255, 0], [255, 255, 85], [255, 255, 170],
          [255, 0, 255], [255, 85, 255], [255, 170, 255],
          [0, 255, 255], [85, 255, 255], [170, 255, 255]]

colors_palette = []
for bgr in colors:
    colors_palette.extend(bgr)


@torch.no_grad()
def export_onnx(onnx_file, model):
    model = model.to(device)
    model.eval()

    img = torch.randn([1, 3, 512, 512], device=device)
    torch.onnx.export(model, (img,), onnx_file, input_names=["img"], dynamic_axes={"img": [0]}, output_names=["out", "out16", "out32", "detail8"], opset_version=12)


def test(image_folder, result_folder, onnx_file="STDC_face_parsing.onnx"):
    session = ort.InferenceSession(onnx_file)

    os.makedirs(result_folder, exist_ok=True)
    files = os.listdir(image_folder)
    for step, file_name in enumerate(files):
        # 1.前处理
        src_file = os.path.join(image_folder, file_name)
        raw: Image.Image = Image.open(src_file).convert("RGB").resize((512, 512))
        img = np.asarray(raw.copy()).astype(np.float32) / 255
        img[:, :, 0] = (img[:, :, 0] - 0.485) / 0.229
        img[:, :, 1] = (img[:, :, 1] - 0.456) / 0.224
        img[:, :, 2] = (img[:, :, 2] - 0.406) / 0.225
        img = img.transpose([2, 0, 1])
        img = np.expand_dims(img, 0)

        # 2.模型推理
        out = session.run(["out"], {"img": img})[0]
        pred = np.argmax(out, 1)[0]  # 注意：一共是20个类别，新增类别`遮挡` ['background', 'skin', 'l_brow', 'r_brow', 'l_eye', 'r_eye', 'eye_g', 'l_ear', 'r_ear', 'ear_r', 'nose', 'mouth', 'u_lip', 'l_lip', 'neck', 'neck_l', 'cloth', 'hair', 'hat', 'occlusion']

        mask = Image.fromarray(pred.astype(np.uint8), "L")
        mask.putpalette(colors_palette, "BGR")
        mask = mask.convert("RGB")

        new_img = Image.blend(raw, mask, 0.4)
        new_img.save(os.path.join(result_folder, file_name))

        print(f"{step + 1}/{len(files)}", end="\r", flush=True)
    print(f"output: {result_folder}")


if __name__ == '__main__':
    # cfg = Config()

    # solver = Solver(cfg)
    # export_onnx("STDC_face_parsing.onnx", solver.model)
    test("/data/face/parsing/dataset/testset_210720_aligned", "/data/STDC_onnx_result_512")
