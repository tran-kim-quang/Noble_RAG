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


def _moving_average(arr: np.ndarray, window_size: int) -> np.ndarray:
    if window_size <= 1 or len(arr) == 0:
        return arr.copy()
    half = window_size // 2
    out = np.empty_like(arr, dtype=np.float32)
    n = len(arr)
    for i in range(n):
        start = max(0, i - half)
        end = min(n, i + half + 1)
        out[i] = np.mean(arr[start:end], axis=0)
    return out


def _stabilize_coords(coord_list, frame_h: int, frame_w: int, window_size: int = 9):
    """
    Stabilize bbox coordinates to reduce frame-to-frame ghosting.
    coords format: (y1, y2, x1, x2)
    """
    if not coord_list:
        return coord_list

    arr = np.asarray(coord_list, dtype=np.float32)
    arr = _moving_average(arr, window_size)

    y1, y2, x1, x2 = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    w = np.maximum(2.0, x2 - x1)
    h = np.maximum(2.0, y2 - y1)

    # Detect jitter and adapt strategy.
    jitter = max(float(np.std(cx)), float(np.std(cy)), float(np.std(w)), float(np.std(h)))
    auto_lock = jitter > 3.0
    mode = (os.getenv("LIVETALKING_COORD_MODE") or "auto").strip().lower()
    size_lock = float(os.getenv("LIVETALKING_COORD_SIZE_LOCK", "0.9"))
    size_lock = float(np.clip(size_lock, 0.0, 1.0))

    if mode == "fixed":
        cx[:] = float(np.median(cx))
        cy[:] = float(np.median(cy))
        w[:] = float(np.median(w))
        h[:] = float(np.median(h))
    elif mode == "lock_size" or (mode == "auto" and auto_lock):
        w_med = float(np.median(w))
        h_med = float(np.median(h))
        w = (1.0 - size_lock) * w + size_lock * w_med
        h = (1.0 - size_lock) * h + size_lock * h_med
        cx = _moving_average(cx, max(3, window_size))
        cy = _moving_average(cy, max(3, window_size))
    # else: smooth-only

    out = []
    for i in range(len(arr)):
        ww = max(2, int(round(w[i])))
        hh = max(2, int(round(h[i])))
        cxx = float(cx[i])
        cyy = float(cy[i])

        nx1 = int(round(cxx - ww / 2))
        ny1 = int(round(cyy - hh / 2))
        nx2 = nx1 + ww
        ny2 = ny1 + hh

        if nx1 < 0:
            nx2 -= nx1
            nx1 = 0
        if ny1 < 0:
            ny2 -= ny1
            ny1 = 0
        if nx2 > frame_w:
            shift = nx2 - frame_w
            nx1 -= shift
            nx2 = frame_w
        if ny2 > frame_h:
            shift = ny2 - frame_h
            ny1 -= shift
            ny2 = frame_h

        nx1 = max(0, nx1)
        ny1 = max(0, ny1)
        nx2 = min(frame_w, max(nx1 + 2, nx2))
        ny2 = min(frame_h, max(ny1 + 2, ny2))
        out.append((ny1, ny2, nx1, nx2))

    logger.info(
        "coords stabilize: mode=%s auto_lock=%s jitter=%.2f window=%d size_lock=%.2f",
        mode, auto_lock, jitter, window_size, size_lock
    )
    return out

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
    model.load_state_dict(new_s)

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
    if frame_list_cycle:
        fh, fw = frame_list_cycle[0].shape[:2]
    else:
        fh, fw = 1080, 1920
    # Stabilize bbox to avoid visible overlay/jitter on avatars with noisy tracking.
    smooth_window = int(os.getenv("LIVETALKING_COORD_SMOOTH_WIN", "11"))
    coord_list_cycle = _stabilize_coords(coord_list_cycle, frame_h=fh, frame_w=fw, window_size=smooth_window)
    input_face_list = glob.glob(os.path.join(face_imgs_path, '*.[jpJP][pnPN]*[gG]'))
    input_face_list = sorted(input_face_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
    face_list_cycle = read_imgs(input_face_list)

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
        img_batch, audiofeat_batch = np.asarray(img_batch), np.asarray(audiofeat_batch)

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
        y1, y2, x1, x2 = bbox
        res_frame = cv2.resize(pred_frame.astype(np.uint8),(x2-x1,y2-y1))
        combine_frame[y1:y2, x1:x2] = res_frame
        return combine_frame
