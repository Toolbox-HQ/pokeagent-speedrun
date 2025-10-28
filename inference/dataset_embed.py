import torch
from transformers import AutoModel
import time
import os 
import json
from pathlib import Path


def main():
    model = AutoModel.from_pretrained("openai/clip-vit-base-patch32")
    
    times = []
   

    for i in range(10):
        img = torch.rand((1,3,224,224))
        s = time.time()
        out = model.extract_image_features(img)
        e = time.time()
        print(e-s)
        times.append(e-s) 

    print(times)       

def split(split_num: int = 500):
    path = ".cache/pokeagent/internet_data"
    files = [str(p) for p in Path(path).iterdir() if p.is_file()]

    lol = []
    l = []
    
    while files:
        l.append(files.pop(0))    
        if len(l) == split_num:
            lol.append(l)
            l = []
    if l:
        lol.append(l)
    
    with open(".cache/pokeagent/video_lol.json", "w") as f:
        json.dump(lol, f)


if __name__ == "__main__":
    split()