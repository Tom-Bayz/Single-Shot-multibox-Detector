#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import numpy as np
import cv2
if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

from ssd import build_ssd
import os

from matplotlib import pyplot as plt
from data import VOCDetection, VOC_ROOT, VOCAnnotationTransform
from data import VOC_CLASSES as labels

import time

import warnings
warnings.simplefilter('ignore')

colors = [(0,0,255),
          (0,0,255),
          (255,0,0),
          (208,0,118),
          (0,255,255),
          (50,255,165),
          (50,255,165)]


# SSDネットワークの定義と重みファイルのロード
net = build_ssd('test', 300, 21)

#r"""
# あんさんぶるスターズ
net.load_weights('./weights/EnsembleStars_SSD.pth')
#net.load_weights('./weights/ensemble_stars_ssd.pth')
video = os.path.join("..","makedata","movie","ensemble_stars02.mov")
#r"""

r"""
# アイドルマスター
net.load_weights('./weights/IdolMaster_SSD.pth')
video = os.path.join("..","makedata","movie","ensemble_stars.mov")
video = os.path.join("..","makedata","movie","idolmaster_test.mov")
r"""


# 動画を読みだす
cap = cv2.VideoCapture(video)

rate = 0
span = 2 #span毎に1回画像を処理する

import matplotlib.pyplot as plt
a = []

start = time.time()
while(cap.isOpened()):



    # 画像キャプチャ
    ret, image = cap.read()

    if not(ret):
        break

    if (rate%span==0):

        # 画像の読み込み
        image = cv2.resize(image , (int(1280*0.5), int(718*0.5)))
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # SSDの入力作成
        x = cv2.resize(image, (300, 300)).astype(np.float32)  # 300*300にリサイズ
        x -= (104.0, 117.0, 123.0)
        x = x.astype(np.float32)
        x = x[:, :, ::-1].copy()
        x = torch.from_numpy(x).permute(2, 0, 1)  # [300,300,3] → [３,300,300]
        xx = Variable(x.unsqueeze(0))     # [3,300,300] → [1,3,300,300]
        if torch.cuda.is_available():
            xx = xx.cuda()

        # 順伝播を実行し、推論結果を出力
        y = net(xx)
        detections = y.data

        # detected objectを画像に合わせてスケーリング
        scale = torch.Tensor(rgb_image.shape[1::-1]).repeat(2)

        # バウンディングボックスとクラス名の表示
        for i in range(detections.size(1)):
            j = 0

            while detections[0,i,j,0] >= 0.5: # 確信度confが0.3以上のボックスを表示
                score = detections[0,i,j,0]


                label_name = labels[i-1]

                display_txt = '%s: %.2f'%(label_name, score)
                pt = (detections[0,i,j,1:]*scale).cpu().numpy()

                s = (int(pt[0]), int(pt[1]))
                h, w = int(pt[2]-pt[0]+1), int(pt[3]-pt[1]+1)

                #print(display_txt,pt)

                #r"""
                color = colors[i]
                cv2.rectangle(image, s, (s[0] + w, s[1] + h), color, 2)
                cv2.putText(img=image,
                            text=display_txt,
                            org=s,
                            fontFace=cv2.FONT_HERSHEY_PLAIN,
                            fontScale=0.8,
                            color=(255,255,255),
                            thickness=1,
                            lineType=cv2.LINE_AA)
                #r"""
                j+=1

        cv2.imshow('frame',image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        pass

    rate += 1
    frame_rate = rate/(time.time()-start)
    print("average frame rate:", round(frame_rate,3) ,"[fps]")
    a.append(frame_rate)

plt.plot(a)
plt.grid(True)
plt.ylabel("frame rate",fontsize=15)
plt.xlabel("frame number",fontsize=15)
plt.show()

cap.release()
cv2.destroyAllWindows()
