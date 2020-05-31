import argparse
import torch
from i3dpt import I3D
import nAction_TF_I3D_maker 
import torch.nn
import os
import cv2
import numpy as np
from torch.utils.data import DataLoader,TensorDataset

dirs = ['0ban','1ban','2ban','3ban']

#画像の画素値とラベルを格納するリスト
data = []
label = []
test_data = []
test_label = []

for i,d in enumerate(dirs):
    #ファイルの取得
    files = os.listdir('./rgb_clips_mp4/'+d)

    for f in files:
        filepath = './rgb_clips_mp4/'
        cap = cv2.VideoCapture(filepath+d+'/'+f)
        vdata = []
        # 動画終了まで繰り返し
        while(cap.isOpened()):
            # フレームを取得
            ret, frame = cap.read()
            if ret == False:
                break
            #-1から1に正規化したい
            # フレームをdataにappendする
            vdata.append(frame)
        data.append(vdata)
        label.append(i)
        
cap.release()

data = np.array(data,dtype='float32')
label = np.array(label,dtype='int64')
train_x = torch.from_numpy(data).float()
train_y = torch.from_numpy(label).long()
train = TensorDataset(train_x,train_y)
train_loader = DataLoader(train,batch_size=2,shuffle=True)
print(train_x.shape)


i3 = I3D(num_classes=5)
i3.eval() #推論モードに切り替える
i3.load_state_dict(torch.load(nAction_TF_I3D_maker/weights/my_RGB_model.ckpt))
i3.train() #訓練モードに切り替える

criterion = torch.nn.NLLLoss()
optimizer = torch.optim.SGD(i3.parameters(), lr=0.001, momentum=0.9)

using_cuda = torch.cuda.is_available()
print(using_cuda)

if using_cuda:
    i3.cuda()
    criterion.cuda()

for epoch in range(4):
    train_loss = 0
    for i, (input_3d,target) in enumerate(train_loader):
        optimizer.zero_grad()
        input_3d_var = input_3d.permute(0,4,1,2,3)
        if using_cuda:
            input_3d_var = input_3d_var.cuda()
        print(input_3d_var.shape)

        out_pt,_ = i3(input_3d_var)
        loss = criterion(out_pt,target)
        train_loss += loss
        loss.backward()
        optimizer.step()
    print("train_epoch:{0}, loss:{1}".format(epoch+1,train_loss))
