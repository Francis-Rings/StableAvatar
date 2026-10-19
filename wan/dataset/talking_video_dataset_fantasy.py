#!/usr/bin/env python3
import math
import os
import random
import warnings
import librosa
import numpy as np
import torch
from PIL import Image
import cv2
from einops import rearrange
import torchvision.transforms.functional as TF
from torch.utils.data.dataset import Dataset
import torch.nn.functional as F


def get_random_mask(shape, image_start_only=False):
    f, c, h, w = shape
    mask = torch.zeros((f, 1, h, w), dtype=torch.uint8)

    if not image_start_only:
        if f != 1:
            mask_index = np.random.choice([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], p=[0.05, 0.2, 0.2, 0.2, 0.05, 0.05, 0.05, 0.1, 0.05, 0.05])
        else:
            mask_index = np.random.choice([0, 1], p = [0.2, 0.8])
        if mask_index == 0:
            center_x = torch.randint(0, w, (1,)).item()
            center_y = torch.randint(0, h, (1,)).item()
            block_size_x = torch.randint(w // 4, w // 4 * 3, (1,)).item()  # 方块的宽度范围
            block_size_y = torch.randint(h // 4, h // 4 * 3, (1,)).item()  # 方块的高度范围

            start_x = max(center_x - block_size_x // 2, 0)
            end_x = min(center_x + block_size_x // 2, w)
            start_y = max(center_y - block_size_y // 2, 0)
            end_y = min(center_y + block_size_y // 2, h)
            mask[:, :, start_y:end_y, start_x:end_x] = 1
        elif mask_index == 1:
            mask[:, :, :, :] = 1
        elif mask_index == 2:
            mask_frame_index = np.random.randint(1, 5)
            mask[mask_frame_index:, :, :, :] = 1
        elif mask_index == 3:
            mask_frame_index = np.random.randint(1, 5)
            mask[mask_frame_index:-mask_frame_index, :, :, :] = 1
        elif mask_index == 4:
            center_x = torch.randint(0, w, (1,)).item()
            center_y = torch.randint(0, h, (1,)).item()
            block_size_x = torch.randint(w // 4, w // 4 * 3, (1,)).item()  # 方块的宽度范围
            block_size_y = torch.randint(h // 4, h // 4 * 3, (1,)).item()  # 方块的高度范围

            start_x = max(center_x - block_size_x // 2, 0)
            end_x = min(center_x + block_size_x // 2, w)
            start_y = max(center_y - block_size_y // 2, 0)
            end_y = min(center_y + block_size_y // 2, h)

            mask_frame_before = np.random.randint(0, f // 2)
            mask_frame_after = np.random.randint(f // 2, f)
            mask[mask_frame_before:mask_frame_after, :, start_y:end_y, start_x:end_x] = 1
        elif mask_index == 5:
            mask = torch.randint(0, 2, (f, 1, h, w), dtype=torch.uint8)
        elif mask_index == 6:
            num_frames_to_mask = random.randint(1, max(f // 2, 1))
            frames_to_mask = random.sample(range(f), num_frames_to_mask)

            for i in frames_to_mask:
                block_height = random.randint(1, h // 4)
                block_width = random.randint(1, w // 4)
                top_left_y = random.randint(0, h - block_height)
                top_left_x = random.randint(0, w - block_width)
                mask[i, 0, top_left_y:top_left_y + block_height, top_left_x:top_left_x + block_width] = 1
        elif mask_index == 7:
            center_x = torch.randint(0, w, (1,)).item()
            center_y = torch.randint(0, h, (1,)).item()
            a = torch.randint(min(w, h) // 8, min(w, h) // 4, (1,)).item()  # 长半轴
            b = torch.randint(min(h, w) // 8, min(h, w) // 4, (1,)).item()  # 短半轴

            for i in range(h):
                for j in range(w):
                    if ((i - center_y) ** 2) / (b ** 2) + ((j - center_x) ** 2) / (a ** 2) < 1:
                        mask[:, :, i, j] = 1
        elif mask_index == 8:
            center_x = torch.randint(0, w, (1,)).item()
            center_y = torch.randint(0, h, (1,)).item()
            radius = torch.randint(min(h, w) // 8, min(h, w) // 4, (1,)).item()
            for i in range(h):
                for j in range(w):
                    if (i - center_y) ** 2 + (j - center_x) ** 2 < radius ** 2:
                        mask[:, :, i, j] = 1
        elif mask_index == 9:
            for idx in range(f):
                if np.random.rand() > 0.5:
                    mask[idx, :, :, :] = 1
        else:
            raise ValueError(f"The mask_index {mask_index} is not define")
    else:
        if f != 1:
            mask[1:, :, :, :] = 1
        else:
            mask[:, :, :, :] = 1
    return mask


class LargeScaleTalkingFantasyVideos(Dataset):
    def __init__(self, txt_path, width, height, n_sample_frames, sample_frame_rate, only_last_features=False, vocal_encoder=None, audio_encoder=None, vocal_sample_rate=16000, audio_sample_rate=24000, enable_inpaint=True, audio_margin=2, vae_stride=None, patch_size=None, wav2vec_processor=None, wav2vec=None):
        self.txt_path = txt_path
        self.width = width
        self.height = height
        self.n_sample_frames = n_sample_frames
        self.sample_frame_rate = sample_frame_rate
        self.only_last_features = only_last_features
        self.vocal_encoder = vocal_encoder
        self.audio_encoder = audio_encoder
        self.vocal_sample_rate = vocal_sample_rate
        self.audio_sample_rate = audio_sample_rate
        self.enable_inpaint = enable_inpaint
        self.wav2vec_processor = wav2vec_processor
        self.audio_margin = audio_margin
        self.vae_stride = vae_stride
        self.patch_size = patch_size
        self.max_area = height * width
        self.aspect_ratio = height / width
        self.video_files = self._read_txt_file_images()

        self.lat_h = round(
            np.sqrt(self.max_area * self.aspect_ratio) // self.vae_stride[1] //
            self.patch_size[1] * self.patch_size[1])
        self.lat_w = round(
            np.sqrt(self.max_area / self.aspect_ratio) // self.vae_stride[2] //
            self.patch_size[2] * self.patch_size[2])

    def _read_txt_file_images(self):
        with open(self.txt_path, 'r') as file:
            lines = file.readlines()
            video_files = []
            for line in lines:
                video_file = line.strip()
                video_files.append(video_file)
        return video_files

    def __len__(self):
        return len(self.video_files)

    def frame_count(self, frames_path):
        files = os.listdir(frames_path)
        png_files = [file for file in files if file.endswith('.png') or file.endswith('.jpg')]
        png_files_count = len(png_files)
        return png_files_count

    def find_frames_list(self, frames_path):
        files = os.listdir(frames_path)
        image_files = [file for file in files if file.endswith('.png') or file.endswith('.jpg')]
        if image_files[0].startswith('frame_'):
            image_files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
        else:
            image_files.sort(key=lambda x: int(x.split('.')[0]))
        return image_files

    def __getitem__(self, idx):
        warnings.filterwarnings('ignore', category=DeprecationWarning)
        warnings.filterwarnings('ignore', category=FutureWarning)

        video_path = os.path.join(self.video_files[idx], "sub_clip.mp4")
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        try:
            is_0_fps = 2 / fps
        except Exception as e:
            print(f"The fps of {video_path} is 0 !!!")
            vocal_audio_path = os.path.join(self.video_files[idx], "audio.wav")
            vocal_duration = librosa.get_duration(filename=vocal_audio_path)
            frames_path = os.path.join(self.video_files[idx], "images")
            total_frame_number = self.frame_count(frames_path)
            fps = total_frame_number / vocal_duration
            print(f"The calculated fps of {video_path} is {fps} !!!")
            # idx = random.randint(0, len(self.video_files) - 1)
            # video_path = os.path.join(self.video_files[idx], "sub_clip.mp4")
            # cap = cv2.VideoCapture(video_path)
            # fps = cap.get(cv2.CAP_PROP_FPS)

        frames_path = os.path.join(self.video_files[idx], "images")

        face_masks_path = os.path.join(self.video_files[idx], "face_masks")
        lip_masks_path = os.path.join(self.video_files[idx], "lip_masks")
        raw_audio_path = os.path.join(self.video_files[idx], "audio.wav")
        # vocal_audio_path = os.path.join(self.video_files[idx], "vocal.wav")
        vocal_audio_path = os.path.join(self.video_files[idx], "audio.wav")
        video_length = self.frame_count(frames_path)
        frames_list = self.find_frames_list(frames_path)

        clip_length = min(video_length, (self.n_sample_frames - 1) * self.sample_frame_rate + 1)

        start_idx = random.randint(0, video_length - clip_length)
        batch_index = np.linspace(
            start_idx, start_idx + clip_length - 1, self.n_sample_frames, dtype=int
        ).tolist()
        all_indices = list(range(0, video_length))
        reference_frame_idx = random.choice(all_indices)

        tgt_pil_image_list = []
        tgt_face_masks_list = []
        tgt_lip_masks_list = []

        # reference_frame_path = os.path.join(frames_path, frames_list[reference_frame_idx])
        reference_frame_path = os.path.join(frames_path, frames_list[start_idx])
        reference_pil_image = Image.open(reference_frame_path).convert('RGB')
        reference_pil_image = reference_pil_image.resize((self.width, self.height))
        reference_pil_image = torch.from_numpy(np.array(reference_pil_image)).float()
        reference_pil_image = reference_pil_image / 127.5 - 1

        for index in batch_index:
            tgt_img_path = os.path.join(frames_path, frames_list[index])
            # file_name = os.path.splitext(os.path.basename(tgt_img_path))[0]
            file_name = os.path.basename(tgt_img_path)
            face_mask_path = os.path.join(face_masks_path, file_name)
            lip_mask_path = os.path.join(lip_masks_path, file_name)
            try:
                tgt_img_pil = Image.open(tgt_img_path).convert('RGB')
            except Exception as e:
                print(f"Fail loading the image: {tgt_img_path}")

            try:
                tgt_lip_mask = Image.open(lip_mask_path)
                # tgt_lip_mask = Image.open(lip_mask_path).convert('RGB')
                tgt_lip_mask = tgt_lip_mask.resize((self.width, self.height))
                tgt_lip_mask = torch.from_numpy(np.array(tgt_lip_mask)).float()
                # tgt_lip_mask = tgt_lip_mask / 127.5 - 1
                tgt_lip_mask = tgt_lip_mask / 255
            except Exception as e:
                print(f"Fail loading the lip masks: {lip_mask_path}")
                tgt_lip_mask = torch.ones(self.height, self.width)
                # tgt_lip_mask = torch.ones(self.height, self.width, 3)
            tgt_lip_masks_list.append(tgt_lip_mask)

            try:
                tgt_face_mask = Image.open(face_mask_path)
                # tgt_face_mask = Image.open(face_mask_path).convert('RGB')
                tgt_face_mask = tgt_face_mask.resize((self.width, self.height))
                tgt_face_mask = torch.from_numpy(np.array(tgt_face_mask)).float()
                tgt_face_mask = tgt_face_mask / 255
                # tgt_face_mask = tgt_face_mask / 127.5 - 1
            except Exception as e:
                print(f"Fail loading the face masks: {face_mask_path}")
                tgt_face_mask = torch.ones(self.height, self.width)
                # tgt_face_mask = torch.ones(self.height, self.width, 3)
            tgt_face_masks_list.append(tgt_face_mask)

            tgt_img_pil = tgt_img_pil.resize((self.width, self.height))
            tgt_img_tensor = torch.from_numpy(np.array(tgt_img_pil)).float()
            tgt_img_normalized = tgt_img_tensor / 127.5 - 1
            tgt_pil_image_list.append(tgt_img_normalized)

        sr = 16000
        vocal_input, sample_rate = librosa.load(vocal_audio_path, sr=sr)
        vocal_duration = librosa.get_duration(filename=vocal_audio_path)
        start_time = batch_index[0] / fps
        end_time = (clip_length / fps) + start_time
        start_sample = int(start_time * sr)
        end_sample = int(end_time * sr)
        try:
            vocal_segment = vocal_input[start_sample:end_sample]
        except:
            print(f"The current vocal segment is too short: {vocal_audio_path}, [{batch_index[0]}, {batch_index[-1]}], fps={fps}, clip_length={clip_length}, vocal_duration={vocal_duration}], [{start_time}, {end_time}]")
            vocal_segment = vocal_input[start_sample:]
        vocal_input_values = self.wav2vec_processor(
            vocal_segment, sampling_rate=sample_rate, return_tensors="pt"
        ).input_values


        tgt_pil_image_list = torch.stack(tgt_pil_image_list, dim=0)
        tgt_pil_image_list = rearrange(tgt_pil_image_list, "f h w c -> f c h w")
        reference_pil_image = rearrange(reference_pil_image, "h w c -> c h w")

        tgt_face_masks_list = torch.stack(tgt_face_masks_list, dim=0)
        tgt_face_masks_list = torch.unsqueeze(tgt_face_masks_list, dim=-1)
        tgt_face_masks_list = rearrange(tgt_face_masks_list, "f h w c -> c f h w")
        tgt_lip_masks_list = torch.stack(tgt_lip_masks_list, dim=0)
        tgt_lip_masks_list = torch.unsqueeze(tgt_lip_masks_list, dim=-1)
        tgt_lip_masks_list = rearrange(tgt_lip_masks_list, "f h w c -> c f h w")


        clip_pixel_values = reference_pil_image.permute(1, 2, 0).contiguous()
        clip_pixel_values = (clip_pixel_values * 0.5 + 0.5) * 255

        cos_similarities = []
        stride = 8
        for i in range(0, tgt_pil_image_list.size()[0] - stride, stride):
            frame1 = tgt_pil_image_list[i]
            frame2 = tgt_pil_image_list[i + stride]
            frame1_flat = frame1.contiguous().view(-1)
            frame2_flat = frame2.contiguous().view(-1)
            cos_sim = F.cosine_similarity(frame1_flat, frame2_flat, dim=0)
            cos_sim = (cos_sim + 1) / 2
            cos_similarities.append(cos_sim.item())
        overall_cos_sim = F.cosine_similarity(tgt_pil_image_list[0].contiguous().view(-1), tgt_pil_image_list[-1].contiguous().view(-1), dim=0)
        overall_cos_sim = (overall_cos_sim + 1) / 2
        cos_similarities.append(overall_cos_sim.item())
        motion_id = (1.0 - sum(cos_similarities) / len(cos_similarities)) * 100


        if "singing" in self.video_files[idx]:
            text_prompt = "The protagonist is singing"
        elif "speech" in self.video_files[idx]:
            text_prompt = "The protagonist is talking"
        elif "dancing" in self.video_files[idx]:
            text_prompt = "The protagonist is simultaneously dancing and singing"
        else:
            text_prompt = ""
            print(1 / 0)

        sample = dict(
            pixel_values=tgt_pil_image_list,
            reference_image=reference_pil_image,
            clip_pixel_values=clip_pixel_values,
            tgt_face_masks=tgt_face_masks_list,
            vocal_input_values=vocal_input_values,
            text_prompt=text_prompt,
            motion_id=motion_id,
            tgt_lip_masks=tgt_lip_masks_list,
            audio_path=raw_audio_path,
        )

        if self.enable_inpaint:
            pixel_value_masks = get_random_mask(tgt_pil_image_list.size(), image_start_only=True)
            masked_pixel_values = tgt_pil_image_list * (1-pixel_value_masks)
            sample["masked_pixel_values"] = masked_pixel_values
            sample["pixel_value_masks"] = pixel_value_masks


        return sample