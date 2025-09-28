from torch.utils.data import Dataset
import os
from util.data import load_json, download_s3_folder, list_files_with_extentions, map_json_to_mp4
from policy import KEY_TO_CLASS, KEY_LIST, CLASS_TO_KEY
import torch 
from torchvision.transforms.functional import resize
from torchcodec.decoders import VideoDecoder
from typing import Tuple

class IDMDataset(Dataset):

    def __init__(self, data_path: str, h=128, w=128, fps: int = 4, s3_bucket: str = None):
        
        self.local_path = os.path.join(".cache", data_path)
        self.fps = fps
        self.h = h
        self.w = w

        if not os.path.isdir(self.local_path):
            download_s3_folder(s3_bucket, data_path, self.local_path)

        # this is slow as data gets large
        self.raw_data = [(load_json(path), map_json_to_mp4(path))\
                          for path in list_files_with_extentions(self.local_path, ".json")]
    
        self.samples = IDMDataset.process_raw_into_samples(self.raw_data, self.fps, 60, 128)
        print('done')

    def __getitem__(self, ind: int) -> Tuple[torch.Tensor, torch.Tensor]:
        (frames, actions, video) = self.samples[ind]
        frames = VideoDecoder(video).get_frames_at(frames.tolist()).data
        return resize(frames, (self.h, self.w)), actions
    
    def __len__(self):
        return len(self.samples)

    @staticmethod
    def action_map(item):
        return torch.tensor([KEY_TO_CLASS[str(i["keys"])] for i in item], dtype=torch.int32)

    @staticmethod
    def process_raw_into_samples(raw_data, sample_fps, video_fps, sample_length):
        samples = []

        frame_step = video_fps // sample_fps

        for item in raw_data:
            video_path = item[1]

            actions = IDMDataset.action_map(item[0])
            indices = torch.arange(start=0, end=len(actions), step=frame_step)[: sample_length * (len(actions) % sample_length)]
            
            # add the batch dimension
            actions = actions[indices].reshape(-1, sample_length)
            indices = indices.reshape(-1, sample_length)
                                     
            samples.extend([(indices[i], actions[i], video_path) for i in range(indices.size()[0])])

        return samples
    
    @staticmethod
    def collate(batch):
        frames, actions =  zip(*batch)
        return torch.stack(frames), torch.stack(actions)


if __name__ == "__main__":
    ds = IDMDataset("pokeagent/emulator_v1", s3_bucket="b4schnei")
    item = ds[0]
    item2 = ds[1]
    x, y = IDM_collator([item, item2])
    print("done")