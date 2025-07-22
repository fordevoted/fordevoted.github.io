import os
import numpy as np
import PIL.Image as Image
import PIL
import cv2
import sys

def provide(in_path, out_root):
    img_out_path = os.path.join(out_root, "image")
    os.makedirs(img_out_path, exist_ok=True)
    pass_flag = False
    for (file, size) in [("color_1920_0864_7680_0864_RGBA8888_0000_000000000.raw", 6635520 - 635520),  # epsilon
                         ("colornoui_1280_0576_5120_0576_RGBA8888_0000_000000000.raw", 2949120 - 49120),
                         ("specular_1280_0576_5120_0576_RGBA8888_0000_000000000.raw", 2949120 - 49120),
                         ("watermask_0640_0288_2560_0288_RGBA8888_0000_000000000.raw", 737280 - 7280)]:
        try:
            if os.path.getsize(os.path.join(in_path, file)) < size:
                pass_flag = True
                break
        except:
            pass_flag = True
            break
    if pass_flag:
        print("broken file!")
        exit(-1)
    for img_path in os.listdir(in_path):
        if img_path.split("_")[0].__contains__("color"):
            with open(os.path.join(in_path, img_path), 'rb') as f:
                raw_data = f.read()
            if img_path.__contains__("noui"):
                arr = np.frombuffer(raw_data, dtype=np.uint8).reshape((576, 1280, 4))
            else:
                arr = np.frombuffer(raw_data, dtype=np.uint8).reshape((864, 1920, 4))
            # arr = arr[..., [2, 1, 0, 3]]  # BGRA -> RGBA
            img = Image.fromarray(arr[:, :, :3], 'RGB')
            img = img.transpose(Image.FLIP_TOP_BOTTOM)

            if img_path.__contains__("noui"):
                img.save(os.path.join(img_out_path, "colorNoUI_" + in_path.split(os.sep)[-1]) + ".png")
                print("processing end for colorNoUI_")
            else:
                img.save(os.path.join(img_out_path, "color_" + in_path.split(os.sep)[-1]) + ".png")
                print("processing end for color_")
        elif img_path.split("_")[0] == "watermask":
            with open(os.path.join(in_path, img_path), 'rb') as f:
                raw_data = f.read()
            # print(img_path)
            try:
                arr = np.frombuffer(raw_data, dtype=np.uint8).reshape((288, 640, 4))
            except:
                arr = np.frombuffer(raw_data, dtype=np.uint8).reshape((27, 64, 4))
            # arr = arr[..., [2, 1, 0, 3]]  # BGRA -> RGBA
            img = Image.fromarray(arr[:, :, :3], 'RGB')
            img = img.resize((1920, 864), resample=PIL.Image.NEAREST)
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
            img = np.sum(img, axis=-1)
            img = np.where(np.array(img).astype(np.float32) > 0, 255, 0).astype(np.uint8)
            img = cv2.imread(os.path.join(img_out_path, "color_" + in_path.split(os.sep)[-1]) + ".png")[:, :,
                  ::-1] & (cv2.resize(img, (1920, 864), cv2.INTER_NEAREST)[:, :, None])
            img = Image.fromarray(img)

            # img = img.convert('L')
            img.save(os.path.join(img_out_path, "watermask_" + in_path.split(os.sep)[-1]) + ".png")
            print("processing end for watermask_")

        elif img_path.split("_")[0] == "specular":
            with open(os.path.join(in_path, img_path), 'rb') as f:
                raw_data = f.read()
            try:
                arr = np.frombuffer(raw_data, dtype=np.uint8).reshape((576, 1280, 4))
            except:
                arr = np.frombuffer(raw_data, dtype=np.uint8).reshape((27, 64, 4))
            # arr = arr[..., [2, 1, 0, 3]]  # BGRA -> RGBA

            img = Image.fromarray(arr[:, :, 3], 'L')
            mask = Image.fromarray(
                np.where(np.array(Image.fromarray(arr[:, :, :3], 'RGB').convert("L")).astype(np.float32) == 0, 255,
                         0).astype(np.uint8))

            img = Image.fromarray(np.where(np.array(img).astype(np.float32) == 4, 255, 0).astype(np.uint8))
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
            mask = mask.transpose(Image.FLIP_TOP_BOTTOM)

            img = np.array(img) & np.array(mask)
            img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
            img = cv2.morphologyEx(img, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
            img = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
            img = cv2.resize(img, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_LINEAR)
            img = cv2.threshold(img, 64, 255, cv2.THRESH_BINARY)[1]
            img = cv2.resize(
                cv2.imread(os.path.join(img_out_path, "color_" + in_path.split(os.sep)[-1]) + ".png")[:, :, ::-1],
                (1280, 576)) & img[:, :, None]
            # img = cv2.morphologyEx(img, cv2.MORPH_ERODE, kernel)
            img = Image.fromarray(img)

            img.save(os.path.join(img_out_path, "specular_" + in_path.split(os.sep)[-1]) + ".png")
            print("processing end for specular_")


if __name__ == '__main__':
    in_path = sys.argv[1]
    # r'/dataset_ns/iHome/DD3/337_dumpData/genshin/water_training_data/img_0717/nvt_2025_07_17_17_52_09_14'
    out_root = sys.argv[2]
    # r'/dataset_ns/iHome/nvt05296/dataset/ADEChallengeData2016/InternImageforGEX/GT_dump/genshin/exp'
    provide(in_path, out_root)
