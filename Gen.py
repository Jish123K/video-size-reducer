import os

import random

import cv2

import moviepy.editor as mp

import skvideo.measure as sk

def convert_and_get_stats(video_name, bitrate, framerate=30):

    source_video_path = 'dataset/videos/'+video_name

    # We should probably be using a ramdisk for this stuff or we're gonna destroy this SSD

    new_video_name = video_name.split('.')[0] + '.webm'

    new_video_path = '/tmp/'+new_video_name

    # Use moviepy to convert the video to webm format

    clip = mp.VideoFileClip(source_video_path)

    clip.write_videofile(new_video_path, bitrate=str(bitrate)+'k', fps=framerate)

    # Use opencv to get the video length and aspect ratio

    cap = cv2.VideoCapture(source_video_path)

    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    h_aspect = float(width) / height

    v_aspect = float(height) / width

    # Use scikit-video to get the video size

    size = os.path.getsize(new_video_path)

    os.remove(new_video_path)

    return size, length/framerate, bitrate, h_aspect, v_aspect

while True:

    video_name = random.choice(os.listdir('dataset/videos'))

    bitrate = random.randrange(10000, 400000)

    size, length, bitrate, h_aspect, v_aspect = convert_and_get_stats(video_name, bitrate)

    dataset.write_data(size, length, bitrate, h_aspect, v_aspect)

