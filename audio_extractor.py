#!/usr/bin/env python3
import os
from moviepy.editor import VideoFileClip
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", type=str)
    parser.add_argument("--saved_audio_path", type=str)
    args = parser.parse_args()
    video_path = args.video_path
    saved_audio_path = args.saved_audio_path
    video = VideoFileClip(video_path)
    audio = video.audio
    audio.write_audiofile(saved_audio_path, codec='pcm_s16le')