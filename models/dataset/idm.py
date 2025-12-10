import torch
from torch.utils.data import Dataset
import os
from models.util.data import load_json, download_s3_folder, list_files_with_extentions, map_json_to_mp4, save_json
from models.policy import KEY_TO_CLASS
import torch 
from torchvision.transforms.functional import resize
from torchcodec.decoders import VideoDecoder
from typing import Tuple,List
import einops
from models.util.data import ValueInterval, apply_video_transform
from joblib import Parallel, delayed, cpu_count
import torch.distributed as dist

def filter_map(l: list):
    return list(filter(lambda x: not "filter" in x.keys() or not x["filter"], l))

class IDMDataset(Dataset):

    def __init__(self,
                 data_path: str,
                 h=128,
                 w=128,
                 fps: int = 4,
                 s3_bucket: str = None,
                 is_val: bool = False,
                 apply_filter: bool= False,
                 buffer_size: int = 60):
        
        self.local_path = data_path
        self.fps = fps
        self.h = h
        self.w = w
        self.buffer_size = buffer_size

        if not os.path.isdir(self.local_path):
            print(f"proxying to s3://{s3_bucket} for {data_path}")
            download_s3_folder(s3_bucket, data_path, self.local_path)

        self.data_files = list_files_with_extentions(self.local_path, ".json")
        self.is_val = is_val

        # this is slow as data gets large
        self.raw_data = [(filter_map(load_json(path)), map_json_to_mp4(path)) for path in self.data_files]

        # filter low action segments
        if apply_filter:
            self.action_filter(self.raw_data)
            self.raw_data = [(filter_map(load_json(path)), map_json_to_mp4(path)) for path in self.data_files]
            print(f"[IDM DATASET] Filtered and reloaded idm data")

        self.samples = IDMDataset.process_raw_into_samples(self.raw_data, self.fps, 60, 128)

    @staticmethod
    def get_frames(video_path, frame_idx, h, w,  is_val=False):

        try:
            frames = VideoDecoder(video_path).get_frames_at(frame_idx).data
        except Exception as e:
            print(f"[VIDEO ERROR] Failed to load {video_path}", flush=True)
            print(f"Idx:\n{frame_idx}", flush=True)
            raise e

        if not is_val:
            frames = apply_video_transform(frames)
        frames = resize(frames, (h, w))

        # IDM expects channel dim is last
        frames = einops.rearrange(frames, "B C H W -> B H W C")
        return frames


    def __getitem__(self, ind: int) -> Tuple[torch.Tensor, torch.Tensor]:
        (frames, actions, video) = zip(*self.samples[ind])
        frames = IDMDataset.get_frames(video[0], frames, self.h, self.w, is_val=self.is_val)

        return frames, torch.tensor(actions, dtype=torch.long)

    def unchanged_interval(vr, start, end, buffer_size):
        diff = vr[end-buffer_size:end] - vr[start].unsqueeze(0)
        return torch.any((diff == 0).all(dim=tuple(range(1, diff.ndim))))

    def action_filter(self, raw_data: List[Tuple]) -> None:

        def process_item(actions, video, json_path):
            try:
                mask = torch.full((len(actions),), False)
                filtered, total = 0, 0

                for (start, end), action in ValueInterval([i["keys"] for i in actions]):
                    total += 1
                    if not action == "none" and IDMDataset.unchanged_interval(VideoDecoder(video), start, end, self.buffer_size):
                        mask[start:end+1] = True
                        filtered += 1
            
                print(f"filtered {100*(filtered / total):.2f}% in {video} : {filtered} / {total} segments")
                assert map_json_to_mp4(json_path) == video, "json path should match mp4 path."

                filtered_data = [
                    item | {"filter": mask[idx].item()} for idx, item in enumerate(actions)
                ]
                save_json(json_path, filtered_data)
            except Exception as e:
                print(f"[IDM Dataset] filter pipeline fail on {video}")
                raise e

        cpu_jobs = cpu_count() // 4
        dist.barrier()

        if not dist.is_initialized() or dist.get_rank() == 0:
            print(f"[IDM DATASET] spawning {cpu_jobs} to filter intervals")
            Parallel(n_jobs=cpu_jobs, backend="threads")(
                delayed(process_item)(actions, video, self.data_files[ind])
                for ind, (actions, video) in enumerate(raw_data)
            )
            
        dist.barrier()

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
    def process_video(video_path, sample_fps, sample_length):
        
        samples = []
        single_sample = []

        vr = VideoDecoder(video_path)
        video_fps = round(vr.metadata.average_fps)
        frame_step = round(video_fps / sample_fps)
        frame_idx = list(range(0, len(vr), frame_step))

        while frame_idx:
            single_sample.append(frame_idx.pop(0))
            if len(single_sample) == sample_length:
                samples.append(single_sample)
                single_sample = []
                                     
        return samples
       

    
    @staticmethod
    def collate(batch):
        frames, actions =  zip(*batch)
        return torch.stack(frames), torch.stack(actions)

if __name__ == "__main__":
    import sys
    assert len(sys.argv) == 2, "Please give the dataset and whether to filter (yes/no)"
    ds = IDMDataset(sys.argv[1], s3_bucket="b4schnei", apply_filter=True, buffer_size = 20)
