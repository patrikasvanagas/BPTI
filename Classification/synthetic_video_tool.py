# -*- coding: utf-8 -*-
"""
Synthetic video tool

Pastes a 4 channel image on top of a background MP4 video and generates movement
assuming straight trajectories. Sources for videos:
https://www.earthcam.com/
https://www.skylinewebcams.com/en.html
X+ direction is to the right, Y+ direction is downwards
np.array(Image.open(PATH)) returns height as the first dimension and length as
the second

Manual parameters:
    FPS : float
        FPS to split the original video and output the result
    ORIGINAL_VIDEO_PATH : string
        Path of the original background .mp4
    DRONE_DIR : string
        Path of the images to be pasted
    SYNTHETIC_FRAMES_DIR : string
        Directory to store the created synthetic frames
    DRONE_INDEX : int
        Index of the image to be pasted in its directory
    INITIAL_RESIZE_RELATIVE : float (0:infty)
        Relative change to the size of the image to be pasted that will be
        resized to simulate movement (if the video is of substantially higher
        resolution than the original pasted image); best to keep the value at 1
    RESIZE_LIMIT_ABSOLUTE : float
        The limit to how much the pasted image can be resized before reversing
        the z-axis movement / 1000
    STARTING_DX_LIMIT : int tuple
        How much off centered the pasted image can be at the start of its
        movement in -x and +x directions relative to the centre of the
        background
    STARTING_DY_LIMIT : int tuple
        How much off centered the pasted image can be at the start of its
        movement in -y and +y directions relative to the centre of the
        background
    SPEED_LIMITS : int tuple > 0 (min, max)
        Limits to the absolute speed of the pasted image per frame
    BLUR_KERNEL : int tuple
        How much the pasted image is blurred
    SECOND_DERIVATIVE : bool
        Turns jitter on or off
    BEGIN_MOVEMENT_AFTER_FRACTION : float [0:1]
        The fraction of the video at the start that will not have images
        pasted
    FRAMES_BEFORE_RESET : int
        How many frames are created before resetting the movement
    FRAMES_BEFORE_RESIZE : int
        How many frames are created before the image is resized for z-axis
        movement simulation
    DRONE_LONG_EDGE_LIMITS : int tuple (min, max)
        Minimum and maximum value of the long edge of the pasted image before
        z-axis movement is reversed
    


Patrikas Vanagas 10/02/2022
"""

import os
from os.path import isfile, join
from random import randint
from random import choice
import re
import numpy as np
from PIL import Image
import cv2
from tqdm import tqdm


# WORKING_DIRECTORY = ''
FPS = 24
ORIGINAL_VIDEO_PATH = "sofia.mp4"
ORIGINAL_FRAMES_DIR = ORIGINAL_VIDEO_PATH.replace(".mp4", "_frames/")
DRONE_DIR = "drones2/"
SYNTHETIC_FRAMES_DIR = "synthetic_frames/"
DRONE_INDEX = 4
INITIAL_RESIZE_RELATIVE = 1
RESIZE_LIMIT_ABSOLUTE = 0.025
STARTING_DX_LIMIT = (-75, 75)
STARTING_DY_LIMIT = (-50, 50)
SPEED_LIMITS = (3, 7)
BLUR_KERNEL = (3, 3)
SECOND_DERIVATIVE = False
BEGIN_MOVEMENT_AFTER_FRACTION = 0.125
FRAMES_BEFORE_RESET = 100
FRAMES_BEFORE_RESIZE = 4
DRONE_LONG_EDGE_LIMITS = (20, 220)


# os.chdir(WORKING_DIRECTORY)

cwd = os.getcwd()


def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split("([0-9]+)", key)]
    return sorted(data, key=alphanum_key)


def calculate_area(xmin, ymin, xmax, ymax):
    return int(xmax - xmin) * (ymax - ymin)


def overlay_image_alpha(img, overlay, x, y, alpha_mask):
    """
    Overlays a 4 channel image over background

    Parameters
    ----------
    img : np.array[0:255]
        Background
    overlay : np.array[0:255]
        4 channel foreground
    x : int
        Abscissas of the left edge to be pasted
    y : int
        Ordinates of the top edge to be pasted
    alpha_mask : np.array
        Alpha channel of the overlay, not normalised to unity

    """
    y1, y2 = max(0, y), min(img.shape[0], y + overlay.shape[0])
    x1, x2 = max(0, x), min(img.shape[1], x + overlay.shape[1])
    y1o, y2o = max(0, -y), min(overlay.shape[0], img.shape[0] - y)
    x1o, x2o = max(0, -x), min(overlay.shape[1], img.shape[1] - x)
    if y1 >= y2 or x1 >= x2 or y1o >= y2o or x1o >= x2o:
        return
    img_crop = img[y1:y2, x1:x2]
    img_overlay_crop = overlay[y1o:y2o, x1o:x2o]
    alpha = alpha_mask[y1o:y2o, x1o:x2o, np.newaxis]
    alpha_inv = 1.0 - alpha
    img_crop[:] = alpha * img_overlay_crop + alpha_inv * img_crop


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


def frame2vid(path_in, fps):
    old_wd = os.getcwd()
    path_out = path_in.strip("/") + ".mp4"
    frame_array = []
    files = [f for f in os.listdir(path_in) if isfile(join(path_in, f))]
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


def video_overlay(
    ORIGINAL_FRAMES_DIR,
    DRONE_DIR,
    SYNTHETIC_FRAMES_DIR,
    DRONE_INDEX,
    INITIAL_RESIZE_RELATIVE,
    RESIZE_LIMIT_ABSOLUTE,
    STARTING_DX_LIMIT,
    STARTING_DY_LIMIT,
    SPEED_LIMITS,
    BLUR_KERNEL,
    SECOND_DERIVATIVE,
    BEGIN_MOVEMENT_AFTER_FRACTION,
    FRAMES_BEFORE_RESET,
    FRAMES_BEFORE_RESIZE,
    DRONE_LONG_EDGE_LIMITS,
):

    original_frames = sorted_alphanumeric(os.listdir(ORIGINAL_FRAMES_DIR))
    drones = sorted_alphanumeric(os.listdir(DRONE_DIR))

    os.makedirs(SYNTHETIC_FRAMES_DIR, exist_ok=True)
    for f in os.listdir(SYNTHETIC_FRAMES_DIR):
        os.remove(os.path.join(SYNTHETIC_FRAMES_DIR, f))
    background_img = np.array(Image.open(ORIGINAL_FRAMES_DIR + original_frames[0]))
    drone_img = np.array(Image.open(DRONE_DIR + "/" + drones[DRONE_INDEX]))
    drone_alpha = drone_img[:, :, 3]
    if INITIAL_RESIZE_RELATIVE != 1:
        drone_img = cv2.resize(
            drone_img,
            (
                int(len(drone_img) * INITIAL_RESIZE_RELATIVE),
                int(len(drone_img[0]) * INITIAL_RESIZE_RELATIVE),
            ),
        )
        drone_alpha = cv2.resize(drone_alpha, (len(drone_img[0]), len(drone_img)),)
    x = int(
        len(background_img) / 2
        + randint(STARTING_DX_LIMIT[0], STARTING_DX_LIMIT[1])
        - len(drone_img)
    )
    y = int(
        len(background_img[0]) / 2
        + randint(STARTING_DY_LIMIT[0], STARTING_DY_LIMIT[1])
        - len(drone_img[0])
    )
    resize_factor = (
        randint(
            1000 - RESIZE_LIMIT_ABSOLUTE * 1000, 1000 + RESIZE_LIMIT_ABSOLUTE * 1000,
        )
        / 1000
    )
    speed = randint(SPEED_LIMITS[0], SPEED_LIMITS[1])
    theta = np.deg2rad(randint(0, 360))
    dx = int(np.ceil(speed * np.cos(speed)))
    dy = int(np.ceil(speed * np.sin(speed)))
    RESIZE_COUNTER = 1

    xmin = np.array([])
    ymin = np.array([])
    xmax = np.array([])
    ymax = np.array([])
    area = np.array([])

    print("Overlaying drone number " + str(DRONE_INDEX))
    for i in tqdm(range(len(original_frames))):

        background_img = np.array(Image.open(ORIGINAL_FRAMES_DIR + original_frames[i]))

        if i > int(len(original_frames) * BEGIN_MOVEMENT_AFTER_FRACTION):
            drone_long_edge = max(len(drone_img), len(drone_img[0]))
            if (
                i == int(len(original_frames) * BEGIN_MOVEMENT_AFTER_FRACTION) + 1
                and i % 4 != 0
            ):
                drone_img = cv2.blur(drone_img, BLUR_KERNEL)
                drone_img[:, :, 3] = drone_alpha
            if i % FRAMES_BEFORE_RESET == 0:
                drone_img = np.array(Image.open(DRONE_DIR + "/" + drones[DRONE_INDEX]))
                drone_alpha = drone_img[:, :, 3]
                if INITIAL_RESIZE_RELATIVE != 1:
                    drone_img = cv2.resize(
                        drone_img,
                        (
                            int(len(drone_img) * INITIAL_RESIZE_RELATIVE),
                            int(len(drone_img[0]) * INITIAL_RESIZE_RELATIVE),
                        ),
                    )
                    drone_alpha = cv2.resize(
                        drone_alpha, (len(drone_img[0]), len(drone_img)),
                    )
                x = int(
                    len(background_img) / 2
                    + randint(STARTING_DX_LIMIT[0], STARTING_DX_LIMIT[1])
                    - len(drone_img)
                )
                y = int(
                    len(background_img[0]) / 2
                    + randint(STARTING_DY_LIMIT[0], STARTING_DY_LIMIT[1])
                    - len(drone_img[0])
                )
                speed = randint(SPEED_LIMITS[0], SPEED_LIMITS[1])
                theta = np.deg2rad(randint(0, 360))
                resize_factor = (
                    randint(
                        1000 - RESIZE_LIMIT_ABSOLUTE * 1000,
                        1000 + RESIZE_LIMIT_ABSOLUTE * 1000,
                    )
                    / 1000
                )
                dx = int(np.ceil(speed * np.cos(theta)))
                dy = int(np.ceil(speed * np.sin(theta)))
                RESIZE_COUNTER = 1
            if drone_long_edge <= DRONE_LONG_EDGE_LIMITS[0]:
                resize_factor = (
                    randint(1000, 1000 + RESIZE_LIMIT_ABSOLUTE * 1000) / 1000
                )
            if drone_long_edge >= DRONE_LONG_EDGE_LIMITS[1]:
                resize_factor = (
                    randint(1000 - RESIZE_LIMIT_ABSOLUTE * 1000, 1000) / 1000
                )
            if i % FRAMES_BEFORE_RESIZE == 0:
                drone_img = np.array(Image.open(DRONE_DIR + "/" + drones[DRONE_INDEX]))
                drone_alpha = drone_img[:, :, 3]
                if INITIAL_RESIZE_RELATIVE != 1:
                    drone_img = cv2.resize(
                        drone_img,
                        (
                            int(len(drone_img) * INITIAL_RESIZE_RELATIVE),
                            int(len(drone_img[0]) * INITIAL_RESIZE_RELATIVE),
                        ),
                    )
                    drone_alpha = cv2.resize(
                        drone_alpha, (len(drone_img[0]), len(drone_img)),
                    )
                drone_img = cv2.resize(
                    drone_img,
                    (
                        int(len(drone_img) * resize_factor ** RESIZE_COUNTER),
                        int(len(drone_img[0]) * resize_factor ** RESIZE_COUNTER),
                    ),
                )
                drone_alpha = cv2.resize(
                    drone_alpha, (len(drone_img[0]), len(drone_img)),
                )
                drone_img = cv2.blur(drone_img, BLUR_KERNEL)
                drone_img[:, :, 3] = drone_alpha
                RESIZE_COUNTER += 1
            if SECOND_DERIVATIVE:
                ddx, ddy = randint(-1, 1), randint(-1, 1)
            else:
                ddx, ddy = 0, 0
            x += dx + ddx
            y += dy + ddy

            if x <= 0:
                # large positive cos, small sin
                theta = np.deg2rad(choice([randint(0, 15), randint(345, 360)]))
                dx = int(np.ceil(speed * np.cos(theta)))
                dy = int(np.ceil(speed * np.sin(theta)))
            if x + len(drone_img[0]) >= len(background_img[0]):
                # large negative cos, small sin
                theta = np.deg2rad(randint(150, 210))
                dx = int(np.ceil(speed * np.cos(theta)))
                dy = int(np.ceil(speed * np.sin(theta)))
            if y <= 0:
                # large negative sin, small cos
                theta = np.deg2rad(randint(255, 285))
                dx = int(np.ceil(speed * np.cos(theta)))
                dy = int(np.ceil(speed * np.sin(theta)))
            if y + len(drone_img) >= len(background_img):
                # large positive sin, small cos
                theta = np.deg2rad(randint(75, 105))
                dx = int(np.ceil(speed * np.cos(theta)))
                dy = int(np.ceil(speed * np.sin(theta)))
            xmin = np.append(xmin, x)
            ymin = np.append(ymin, y)
            xmax = np.append(xmax, x + len(drone_img[0]))
            ymax = np.append(ymax, y + len(drone_img))
            alpha_mask = drone_img[:, :, 3] / 255.0
            img_result = background_img[:, :, :3].copy()
            img_overlay = drone_img[:, :, :3]
            overlay_image_alpha(img_result, img_overlay, x, y, alpha_mask)
            Image.fromarray(img_result).save(SYNTHETIC_FRAMES_DIR + "/%d.jpg" % i)
        else:
            xmin = np.append(xmin, None)
            ymin = np.append(ymin, None)
            xmax = np.append(xmax, None)
            ymax = np.append(ymax, None)
            Image.fromarray(background_img).save(SYNTHETIC_FRAMES_DIR + "/%d.jpg" % i)
    for m in range(len(xmin)):
        if xmin[m] == None:
            area = np.append(area, None)
        else:
            area = np.append(area, calculate_area(xmin[m], ymin[m], xmax[m], ymax[m]),)
    bboxes = np.stack((xmin, ymin, xmax, ymax, area), axis=1)
    np.savetxt("bboxes.txt", bboxes, fmt="%s")


def main():
    vid2frame(ORIGINAL_VIDEO_PATH)
    video_overlay(
        ORIGINAL_FRAMES_DIR,
        DRONE_DIR,
        SYNTHETIC_FRAMES_DIR,
        DRONE_INDEX,
        INITIAL_RESIZE_RELATIVE,
        RESIZE_LIMIT_ABSOLUTE,
        STARTING_DX_LIMIT,
        STARTING_DY_LIMIT,
        SPEED_LIMITS,
        BLUR_KERNEL,
        SECOND_DERIVATIVE,
        BEGIN_MOVEMENT_AFTER_FRACTION,
        FRAMES_BEFORE_RESET,
        FRAMES_BEFORE_RESIZE,
        DRONE_LONG_EDGE_LIMITS,
    )
    frame2vid(SYNTHETIC_FRAMES_DIR, FPS)


if __name__ == "__main__":
    main()
