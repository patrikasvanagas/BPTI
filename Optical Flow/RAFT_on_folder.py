import shutil
import os
import re
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import PIL
import torch
import torchvision.transforms.functional as F
import torchvision.transforms as T
import torchvision
from torchvision.utils import flow_to_image
from torchvision.utils import save_image
from torchvision.models.optical_flow import raft_large
from torchvision.io import read_video
from tqdm import tqdm
import tempfile
from pathlib import Path
from urllib.request import urlretrieve
import ffmpeg

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model = raft_large(pretrained=True, progress=False).to(DEVICE)
model = model.eval()
plt.rcParams["savefig.bbox"] = "tight"


def preprocess(batch):
    # image dimension must be divisible by 8
    transforms = T.Compose(
        [
            T.ConvertImageDtype(torch.float32),
            T.Normalize(mean=0.5, std=0.5),  # map [0, 1] into [-1, 1]
            T.Resize(size=(HEIGHT, WIDTH)),
        ]
    )
    batch = transforms(batch)
    return batch


def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split("([0-9]+)", key)]
    return sorted(data, key=alphanum_key)


def vid2frame(path_in):
    old_wd = os.getcwd()
    vidcap = cv2.VideoCapture(path_in)
    # success, image = vidcap.read()
    image = vidcap.read()[1]
    frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    path_out = path_in.replace(".mp4", "_frames/")
    os.makedirs(path_out, exist_ok=True)
    for o in os.listdir(path_out):
        os.remove(os.path.join(path_out, o))
    os.chdir(path_out)
    print("Converting " + path_in + " to frames:")
    for m in tqdm(range(frames)):
        cv2.imwrite("%d.jpg" % m, image)
        # success, image = vidcap.read()
        image = vidcap.read()[1]
    vidcap.release()
    os.chdir(old_wd)


def frame2vid(path_in, path_out, fps):
    old_wd = os.getcwd()
    # path_out = path_in.strip("/") + ".mp4"
    frame_array = []
    files = [f for f in os.listdir(path_in) if os.path.isfile(os.path.join(path_in, f))]
    files = sorted_alphanumeric(files)
    print("Reading " + path_in + " for conversion to video:")
    for m in tqdm(range(len(files))):
        filename = path_in + files[m]
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width, height)
        frame_array.append(img)
    out = cv2.VideoWriter(path_out, cv2.VideoWriter_fourcc(*"MP4V"), fps, size)
    print("Converting " + path_in + " to video:")
    for m in tqdm(range(len(frame_array))):
        out.write(frame_array[m])
    out.release()
    os.chdir(old_wd)


fps_96_list = [
    "boat7",
    "boat8",#...
]

SEQUENCE_FOLDER = "/home/patrikas_v/videos/"
RESULTS_FOLDER = "/home/patrikas_v/results/"
os.makedirs(RESULTS_FOLDER, exist_ok=True)
sequences = os.listdir(SEQUENCE_FOLDER)

for i in tqdm(range(len(sequences))):
    sequence = sequences[i]
    if sequence in fps_96_list:
        FPS = 96
    else:
        FPS = 30
    input_video_path = SEQUENCE_FOLDER + sequence
    output_photos_folder = RESULTS_FOLDER + sequence.strip(".mp4") + "_flow/"
    os.makedirs(output_photos_folder, exist_ok=True)
    output_tensor_path = RESULTS_FOLDER + sequence.strip(".mp4") + "_flow.pt"
    output_video_path = RESULTS_FOLDER + sequence.strip(".mp4") + "_flow.mp4"

    frames, list_of_flows, predicted_flows, flow_imgs, grid = [], [], [], [], []
    frames, _, _ = read_video(
        input_video_path, start_pts=0, end_pts=9999, pts_unit="sec"
    )
    frames = frames.permute(0, 3, 1, 2)
    HEIGHT = frames.size()[2]
    WIDTH = frames.size()[3]
    num_frames = len(frames)
    indices = np.arange(0, num_frames)
    full_flows_stack = torch.empty([(len(indices) - 2), 2, HEIGHT, WIDTH])
    torch.cuda.empty_cache()
    print("Calculating flow for", sequence)
    for i in tqdm(indices[0:-2]):
        img_1 = frames[i : (i + 1)]
        img_2 = frames[(i + 1) : (i + 2)]
        img_1_preprocessed = preprocess(img_1).to(DEVICE)
        img_2_preprocessed = preprocess(img_2).to(DEVICE)
        with torch.no_grad():
            list_of_flows = model(
                img_1_preprocessed.to(DEVICE), img_2_preprocessed.to(DEVICE)
            )
        flow = list_of_flows[-1][0]
        full_flows_stack[i] = flow
        flow_img = flow_to_image(flow)
        (F.to_pil_image(flow_img)).save(output_photos_folder + "/" + str(i) + ".jpg")
    torch.save(full_flows_stack, output_tensor_path)
    frame2vid(output_photos_folder, output_video_path, FPS)
    shutil.rmtree(output_photos_folder)
    frame2vid(output_photos_folder, output_video_path, FPS)
    shutil.rmtree(output_photos_folder)
    shutil.rmtree(output_photos_folder)
