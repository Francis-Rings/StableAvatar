#!/usr/bin/env python3
import argparse
import os

import cv2
import mediapipe as mp
import numpy as np

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--folder_root", type=str)
    parser.add_argument("--start", type=int, help="Specify the value of start")
    parser.add_argument("--end", type=int, help="Specify the value of end")
    args = parser.parse_args()

    folder_root = args.folder_root
    start = args.start
    end = args.end

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=10)

    upper_lip_idx = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291]
    lower_lip_idx = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291]

    for idx in range(start, end):
        subfolder = str(idx).zfill(5)
        subfolder_path = os.path.join(folder_root, subfolder)
        images_folder = os.path.join(subfolder_path, "images")
        if os.path.exists(images_folder):
            face_masks_folder = os.path.join(subfolder_path, "lip_masks")
            os.makedirs(face_masks_folder, exist_ok=True)
            for root, dirs, files in os.walk(images_folder):
                for file in files:
                    if file.endswith('.png'):
                        file_name = os.path.splitext(file)[0]
                        image_name = file_name + '.png'
                        image_legal_path = os.path.join(images_folder, image_name)
                        if os.path.exists(os.path.join(face_masks_folder, file_name + '.png')):
                            existed_path = os.path.join(face_masks_folder, file_name + '.png')
                            print(f"{existed_path} already exists!")
                            continue

                        face_save_path = os.path.join(face_masks_folder, file_name + '.png')

                        image = cv2.imread(image_legal_path)
                        h, w, _ = image.shape
                        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        results = face_mesh.process(rgb_image)
                        mask = np.zeros((h, w), dtype=np.uint8)

                        if results.multi_face_landmarks:
                            for face_landmarks in results.multi_face_landmarks:
                                upper_points = np.array([
                                    [int(face_landmarks.landmark[i].x * w), int(face_landmarks.landmark[i].y * h)]
                                    for i in upper_lip_idx
                                ], dtype=np.int32)
                                lower_points = np.array([
                                    [int(face_landmarks.landmark[i].x * w), int(face_landmarks.landmark[i].y * h)]
                                    for i in lower_lip_idx
                                ], dtype=np.int32)
                                cv2.fillPoly(mask, [upper_points], 255)
                                cv2.fillPoly(mask, [lower_points], 255)
                        else:
                            print(f"No face detected in {image_legal_path}. Saving empty mask.")
                        cv2.imwrite(face_save_path, mask)
                        print(f"Lip mask saved to {face_save_path}")
        else:
            print(f"{images_folder} does not exist")
            continue