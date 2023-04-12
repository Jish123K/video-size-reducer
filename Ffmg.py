import os

import subprocess

import re

def get_video_aspect_ratio(video_path):

    result = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries',

                             'stream=display_aspect_ratio', '-of', 'default=noprint_wrappers=1:nokey=1', video_path],

                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)

    if result.returncode != 0:

        raise ValueError('Error while getting aspect ratio: ' + result.stderr.strip())

    horizontal, vertical = map(float, result.stdout.strip().split(':'))

    return horizontal, vertical

def get_video_length(video_path):

    result = subprocess.run(['ffprobe', '-v', 'error', '-show_entries', 'format=duration',

                             '-of', 'default=noprint_wrappers=1:nokey=1', video_path],

                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)

    if result.returncode != 0:

        raise ValueError('Error while getting video length: ' + result.stderr.strip())

    return float(result.stdout.strip())

def convert_to_webm(video_path, bitrate, output_video_path=None, framerate=30):

    if output_video_path is None:

        output_video_path = re.sub("\..*$",".webm",video_path)

    result = subprocess.run(['ffmpeg', '-i', video_path, '-preset', 'ultrafast', '-vcodec', 'libvpx', '-vf',

                             'scale=480:-1', '-r', str(framerate), '-b:v', str(bitrate), output_video_path],

                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)

    if result.returncode != 0:

        raise ValueError('Error while converting video to WebM: ' + result.stderr.strip())

    filesize = os.path.getsize(output_video_path)

    return output_video_path, filesize

