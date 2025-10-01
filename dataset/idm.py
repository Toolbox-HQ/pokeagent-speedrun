import torch
from torch.utils.data import Dataset
import os
from util.data import load_json, download_s3_folder, list_files_with_extentions, map_json_to_mp4
from policy import KEY_TO_CLASS
import torch 
from torchvision.transforms.functional import resize
from torchcodec.decoders import VideoDecoder
from typing import Tuple,List
import einops
from util.data import ValueInterval

class IDMDataset(Dataset):

    def __init__(self, data_path: str, h=128, w=128, fps: int = 4, s3_bucket: str = None, apply_filter = True):
        
        self.local_path = os.path.join(".cache", data_path)
        self.fps = fps
        self.h = h
        self.w = w

        if not os.path.isdir(self.local_path):
            download_s3_folder(s3_bucket, data_path, self.local_path)

        # this is slow as data gets large
        self.raw_data = [(load_json(path), map_json_to_mp4(path))\
                          for path in list_files_with_extentions(self.local_path, ".json")]

        IDMDataset.action_filter(self.action_filter(self.raw_data))
        self.samples = IDMDataset.process_raw_into_samples(self.raw_data, self.fps, 60, 128)

    def __getitem__(self, ind: int) -> Tuple[torch.Tensor, torch.Tensor]:
        (frames, actions, video) = zip(*self.samples[ind])
        frames = VideoDecoder(video[0]).get_frames_at(frames).data
        frames = resize(frames, (self.h, self.w))

        # IDM expects channel dim is last
        return einops.rearrange(frames, "B C H W -> B H W C"), torch.tensor(actions, dtype=torch.long)

    def unchanged_interval(vr, start, end):
        buffer = 60
        diff = vr[end-buffer:end] - vr[start].unsqueeze(0)
        return torch.any((diff == 0).all(dim=tuple(range(1, diff.ndim))))

    @staticmethod
    def action_filter(raw_data: List[Tuple])-> None:
       """
       Filters actions where the game remained static
       over the entire course of the action.
       """
       for ind, (actions, video) in enumerate(raw_data):
            mask = torch.full((len(actions),), False)

            for start, end in ValueInterval([i["keys"] for i in actions]):
                print(f"{start} : {end} : {video}")
                vr = VideoDecoder(video)
                if IDMDataset.unchanged_interval(vr, start, end):
                    mask[start : end] = True
                    print("filtered")
                    
            raw_data[ind] = ([x for x, m in zip(raw_data[ind][0], mask) if not m], raw_data[ind][1])

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def process_raw_into_samples(raw_data, sample_fps, video_fps, sample_length):
        
        samples = []
        single_sample = []

        frame_step = round(video_fps / sample_fps)

        for item in raw_data:
            video_path = item[1]
            frame_data = item[0]

            frame_data = [(frame_data[i]["frame"], KEY_TO_CLASS[frame_data[i]["keys"]], video_path) \
                        for i in range(0, len(frame_data), frame_step)]

            while frame_data:
                single_sample.append(frame_data.pop(0))
                if len(single_sample) == sample_length:
                    samples.append(single_sample)
                    single_sample = []
                                     
        return samples
    
    @staticmethod
    def collate(batch):
        frames, actions =  zip(*batch)
        return torch.stack(frames), torch.stack(actions)


if __name__ == "__main__":
    ds = IDMDataset("pokeagent/emulator_v1", s3_bucket=None)

    print("done")