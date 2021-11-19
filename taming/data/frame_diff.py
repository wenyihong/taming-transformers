import os
import json
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, IterableDataset
import random
import cv2


class FrameDiffDataset(Dataset):
    def __init__(self, dataroot=""):
        self.dataroot = dataroot
        frames = os.listdir(dataroot)
        clips = [ '_'.join(x.split('_')[:-1]) for x in frames]
        self.clips = random.Random(42).shuffle(list(set(clips)))
        self.frames_per_clip = 16
    
    def __len__(self):
        return len(self.clips)*self.frames_per_clip
    
    def __getitem__(self, idx):
        clipidx = int(idx/self.frames_per_clip)
        frameidx = idx % self.frames_per_clip
        frame1_path = self.clips[clipidx]+'_'+str(frameidx).zfill(4)+'.jpg'
        frame1_path = os.path.join(self.dataroot, frame1_path)
        frame2_path = self.clips[clipidx]+'_'+str(frameidx+1).zfill(4)+'.jpg'
        frame2_path = os.path.join(self.dataroot, frame2_path)
        frame_paths = [frame1_path, frame2_path]
        frames = []
        for i in range(2):
            fr = Image.open(frame_paths[i])
            fr = np.array(fr).astype(np.uint8)
            fr = (fr/127.5 -1.0).astype(np.float32)
            frames.append(torch.from_numpy(fr).unsqueeze(0))
        
        return torch.cat(frames, dim=0)
    
    
import tarfile

class FrameDiffTarDataset(Dataset):
    def __init__(self, tarpath=""):
        self.tarpath = tarpath
        self.tar = tarfile.open(tarpath, 'r', bufsize=16*1024*1024) 
        self.namelist = self.tar.getnames()
        # print(self.namelist)
        raw_clip_list = ['/'.join(x.split('/')[:-1]) for x in self.namelist]
        self.cliplist = list(set(raw_clip_list))        
        random.Random(42).shuffle(self.cliplist)
        self.frames_per_clip = 16 # 17 actually, with 16 framepairs

    def __len__(self):
        return len(self.cliplist)*self.frames_per_clip
    
    def __getitem__(self, idx):
        clipidx = int(idx/self.frames_per_clip)
        frameidx = idx % self.frames_per_clip
        frame1_path = self.cliplist[clipidx]+'/frame_'+str(frameidx).zfill(4)+'.jpg'
        frame2_path = self.cliplist[clipidx]+'/frame_'+str(frameidx+1).zfill(4)+'.jpg'
        frame_paths = [frame1_path, frame2_path]
        frames = []
        for i in range(2):
            # print("PATH", frame_paths[i])
            fp = self.tar.extractfile(frame_paths[i])
            # print("FP READ", fp)
            fr = Image.open(fp)
            fr = np.array(fr).astype(np.uint8)
            fr = (fr/127.5 -1.0).astype(np.float32)
            frames.append(torch.from_numpy(fr).unsqueeze(0))
        
        return {'image': torch.cat(frames, dim=0)}

class FrameDiffTarIterDataset(IterableDataset):
    def __init__(self, tarpath="", diff_filter=None):
        self.tarpath = tarpath
        self.tar = tarfile.open(tarpath, 'r', bufsize=16*1024*1024) 
        self.frames_per_clip = 16 # 17 actually, with 16 framepairs
        self.endframe_idx = str(self.frames_per_clip).zfill(4)
        self.last_frame = None
        self.cnt = 0
        self.tar.close()
        self.tar = None
        self.diff_filter = diff_filter
    
    def __iter__(self):
        if self.tar is None:
            self.tar = tarfile.open(self.tarpath, 'r', bufsize=16*1024*1024) 
        for info in self.tar:
            self.cnt += 1
            file = tarfile.ExFileObject(self.tar, info)
            tmp_frame = Image.open(file)
            cv2_tmp_frame = cv2.cvtColor(np.asarray(tmp_frame),cv2.COLOR_RGB2BGR)
            
            tmp_frame = np.array(tmp_frame).astype(np.uint8)
            tmp_frame = (tmp_frame/127.5 -1.0).astype(np.float32)
            tmp_frame = torch.from_numpy(tmp_frame).unsqueeze(0)
            
            if self.last_frame is None:
                assert info.name.split('/')[-1] == 'frame_0000.jpg'
                self.last_frame = tmp_frame
                continue
            
            # check difference
            if self.diff_filter is not None:
                frame_diff = torch.abs(tmp_frame-self.last_frame)
                diff_mean = torch.mean(frame_diff, dim=[-1, -2, -3])
                if diff_mean <= self.diff_filter:
                    if info.name.split('/')[-1] == f'frame_{self.endframe_idx}.jpg':
                        self.last_frame = None
                        continue
                    else:
                        self.last_frame = tmp_frame
                        continue
            
            # cal gradient
            grad_tmp_frame_x = cv2.Sobel(cv2_tmp_frame, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=-1)
            grad_tmp_frame_y = cv2.Sobel(cv2_tmp_frame, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=-1)
            cv2_grad_map = cv2.addWeighted(cv2.convertScaleAbs(grad_tmp_frame_x), 0.5, cv2.convertScaleAbs(grad_tmp_frame_y), 0.5, 0)
            PIL_grad_map = Image.fromarray(cv2.cvtColor(cv2_grad_map,cv2.COLOR_BGR2RGB))
            PIL_grad_map = np.array(PIL_grad_map).astype(np.uint8)
            grad_map = np.array(PIL_grad_map/255.0).astype(np.float32)
            grad_map = torch.from_numpy(grad_map)
            

            
            self.now_frame = tmp_frame
            self.now_grad_map = grad_map
            yield {'image': torch.cat((self.last_frame, self.now_frame), dim=0),
                   'grad': grad_map}
            if info.name.split('/')[-1] == f'frame_{self.endframe_idx}.jpg':
                self.last_frame = None
                assert self.cnt % (self.frames_per_clip+1) == 0
            else:
                self.last_frame = self.now_frame
            