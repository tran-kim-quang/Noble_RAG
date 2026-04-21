###############################################################################
#  Copyright (C) 2024 LiveTalking@lipku https://github.com/lipku/LiveTalking
#  email: lipku@foxmail.com
# 
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#  
#       http://www.apache.org/licenses/LICENSE-2.0
# 
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
###############################################################################
#
#  Wav2Lip 数字人 — 迁移自 lipreal.py + lipasr.py
#

import math
import torch
import numpy as np

import os
import time
import cv2
import glob
import pickle
import copy

import queue
from queue import Queue
from threading import Thread, Event
import torch.multiprocessing as mp

from avatars.audio_features.mel import MelASR
import asyncio
from av import AudioFrame, VideoFrame
from avatars.wav2lip.models import Wav2Lip
from avatars.base_avatar import BaseAvatar

from tqdm import tqdm
from utils.logger import logger
from utils.image import read_imgs, mirror_index
from utils.device import initialize_device
from registry import register

device = initialize_device()
logger.info('Using {} for inference.'.format(device))

def _load(checkpoint_path):
    if device == 'cuda':
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path,
                                map_location=lambda storage, loc: storage)
    return checkpoint

def load_model(path):
    model = Wav2Lip()
    logger.info("Load checkpoint from: {}".format(path))
    checkpoint = _load(path)
    s = checkpoint["state_dict"]
    new_s = {}
    for k, v in s.items():
        new_s[k.replace('module.', '')] = v
    model.load_state_dict(new_s, strict=False)

    model = model.to(device)
    return model.eval()

def load_avatar(avatar_id):
    avatar_path = f"./data/avatars/{avatar_id}"
    full_imgs_path = f"{avatar_path}/full_imgs" 
    face_imgs_path = f"{avatar_path}/face_imgs" 
    coords_path = f"{avatar_path}/coords.pkl"
    
    with open(coords_path, 'rb') as f:
        coord_list_cycle = pickle.load(f)
    frame_list_cycle = None
    input_img_list = glob.glob(os.path.join(full_imgs_path, '*.[jpJP][pnPN]*[gG]'))
    input_img_list = sorted(input_img_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
    frame_list_cycle = read_imgs(input_img_list)
    input_face_list = glob.glob(os.path.join(face_imgs_path, '*.[jpJP][pnPN]*[gG]'))
    input_face_list = sorted(input_face_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
    face_list_cycle = read_imgs(input_face_list)
    # wav2lip_v2 in this repo expects a high-res face crop (>=256) to keep
    # downsampling stages valid. Older avatars may contain 96x96 crops.
    target_face_size = int((os.getenv("WAV2LIP_FACE_SIZE") or "256").strip())
    if face_list_cycle:
        h, w = face_list_cycle[0].shape[:2]
        if h != target_face_size or w != target_face_size:
            logger.warning(
                "Avatar face crop size is %sx%s, auto-resizing to %sx%s for wav2lip_v2 compatibility.",
                h,
                w,
                target_face_size,
                target_face_size,
            )
            face_list_cycle = [
                cv2.resize(face, (target_face_size, target_face_size), interpolation=cv2.INTER_LINEAR)
                for face in face_list_cycle
            ]

    return frame_list_cycle,face_list_cycle,coord_list_cycle

@torch.no_grad()
def warm_up(batch_size,model,modelres):
    # 预热函数
    logger.info('warmup model...')
    img_batch = torch.ones(batch_size, 6, modelres, modelres).to(device)
    mel_batch = torch.ones(batch_size, 1, 80, 16).to(device)
    model(mel_batch, img_batch)

@register("avatar", "wav2lip")
class LipReal(BaseAvatar):
    @torch.no_grad()
    def __init__(self, opt, model, avatar):
        super().__init__(opt)

        #self.fps = opt.fps # 20 ms per frame
        
        # self.batch_size = opt.batch_size
        # self.idx = 0
        # self.res_frame_queue = Queue(self.batch_size*2)
        self.model = model

        self.frame_list_cycle,self.face_list_cycle,self.coord_list_cycle = avatar

        self.asr = MelASR(opt,self)
        self.asr.warm_up()
    
    def inference_batch(self, index, audiofeat_batch):
        # 这里的 index 是针对当前 avatar 的索引
        # 返回一个 batch 的推理结果，batch 大小由 self.batch_size 决定
        length = len(self.face_list_cycle)
        img_batch = []
        for i in range(self.batch_size):
            idx = mirror_index(length, index + i)
            face = self.face_list_cycle[idx]
            img_batch.append(face)
        img_batch = np.asarray(img_batch)

        # Keep mel batch shape stable for Wav2Lip: [B, 80, 16].
        normalized_audio = []
        if audiofeat_batch is None:
            audiofeat_batch = []
        for feat in audiofeat_batch:
            arr = np.asarray(feat, dtype=np.float32)
            if arr.ndim == 1:
                arr = np.reshape(arr, (80, -1)) if arr.size >= 80 else np.zeros((80, 16), dtype=np.float32)
            if arr.ndim != 2:
                arr = np.squeeze(arr)
                if arr.ndim != 2:
                    arr = np.zeros((80, 16), dtype=np.float32)
            # Force 80 mel bins.
            if arr.shape[0] != 80:
                if arr.shape[1] == 80:
                    arr = arr.T
                else:
                    fixed = np.zeros((80, arr.shape[1] if arr.ndim == 2 and arr.shape[1] > 0 else 16), dtype=np.float32)
                    rows = min(80, arr.shape[0]) if arr.ndim == 2 else 0
                    if rows > 0:
                        fixed[:rows, :arr.shape[1]] = arr[:rows, :]
                    arr = fixed
            # Force 16 time steps.
            t = arr.shape[1]
            if t < 16:
                arr = np.pad(arr, ((0, 0), (0, 16 - t)), mode="constant")
            elif t > 16:
                arr = arr[:, :16]
            normalized_audio.append(arr)

        # Ensure audio batch size matches image batch size.
        while len(normalized_audio) < self.batch_size:
            normalized_audio.append(np.zeros((80, 16), dtype=np.float32))
        if len(normalized_audio) > self.batch_size:
            normalized_audio = normalized_audio[:self.batch_size]

        audiofeat_batch = np.asarray(normalized_audio, dtype=np.float32)

        img_masked = img_batch.copy()
        img_masked[:, face.shape[0]//2:] = 0

        img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
        audiofeat_batch = np.reshape(audiofeat_batch, [len(audiofeat_batch), audiofeat_batch.shape[1], audiofeat_batch.shape[2], 1])
        
        img_batch = torch.FloatTensor(np.transpose(img_batch, (0, 3, 1, 2))).to(device)
        audiofeat_batch = torch.FloatTensor(np.transpose(audiofeat_batch, (0, 3, 1, 2))).to(device)

        with torch.no_grad():
            pred = self.model(audiofeat_batch, img_batch)
        pred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.
        return pred

    def paste_back_frame(self,pred_frame,idx:int):
        bbox = self.coord_list_cycle[idx]
        combine_frame = copy.deepcopy(self.frame_list_cycle[idx])
        y1, y2, x1, x2 = [int(v) for v in bbox]
        target_w = x2 - x1
        target_h = y2 - y1
        if target_w <= 0 or target_h <= 0:
            return combine_frame

        res_frame = cv2.resize(pred_frame.astype(np.uint8), (target_w, target_h))
        frame_h, frame_w = combine_frame.shape[:2]

        dx1 = max(0, x1)
        dy1 = max(0, y1)
        dx2 = min(frame_w, x2)
        dy2 = min(frame_h, y2)
        if dx1 >= dx2 or dy1 >= dy2:
            return combine_frame

        sx1 = dx1 - x1
        sy1 = dy1 - y1
        sx2 = sx1 + (dx2 - dx1)
        sy2 = sy1 + (dy2 - dy1)
        combine_frame[dy1:dy2, dx1:dx2] = res_frame[sy1:sy2, sx1:sx2]
        return combine_frame

