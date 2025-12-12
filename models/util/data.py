import os
import orjson
import re 
import os.path as path
from typing import Dict, List
import random

def sample_transform_params():
    hue = random.uniform(-0.2, 0.2)
    saturation = random.uniform(0.8, 1.2)
    brightness = random.uniform(0.8, 1.2)
    contrast = random.uniform(0.8, 1.2)
    angle = random.uniform(-2, 2)
    scale = random.uniform(0.98, 1.02)
    shear = random.uniform(-2, 2)
    translate_x = random.uniform(-2, 2)
    translate_y = random.uniform(-2, 2)
    return {
        "hue": hue,
        "saturation": saturation,
        "brightness": brightness,
        "contrast": contrast,
        "angle": angle,
        "scale": scale,
        "shear": shear,
        "translate": (translate_x, translate_y),
    }

def apply_video_transform(img):
    import torchvision.transforms.functional as F
    B, C, W, H = img.shape
    assert C == 3, "Channel dim must be first"

    params = sample_transform_params()
    x = F.adjust_hue(img, params["hue"])
    x = F.adjust_saturation(x, params["saturation"])
    x = F.adjust_brightness(x, params["brightness"])
    x = F.adjust_contrast(x, params["contrast"])
    x = F.affine(
        x,
        angle=params["angle"],
        translate=params["translate"],
        scale=params["scale"],
        shear=params["shear"],
    )
    return x


def reduce_dict(l: List) -> Dict:
    rd = {}
    for d in l:
        for k in d.keys():
            if not k in rd:
                rd[k] = (d[k],1)
            else:
                (x, y) = rd[k]
                rd[k] = (x+d[k] ,y+1)
    return {k : x/y for k, (x,y) in rd.items()}

class ValueInterval():

    def __init__(self, list):
        self.curr_idx = 0
        self.list = list

    def __iter__(self):
        return self

    def __next__(self):

        if not self.curr_idx < len(self.list):
            raise StopIteration

        start_idx = self.curr_idx
        item = self.list[start_idx]
        self.curr_idx += 1

        while self.curr_idx < len(self.list) and self.list[self.curr_idx] == item:
            self.curr_idx += 1

        return (start_idx, self.curr_idx-1), item

def list_files_with_extentions(dir: str, ext: str):
    matches = []
    for root, _, files in os.walk(dir):
        for name in files:
            if name.endswith(ext):
                matches.append(os.path.join(root, name))
    return matches

def map_json_to_mp4(filename):
    dirname, basename = os.path.split(filename)
    mp4_path = path.join(dirname, re.sub(r'\.json$', '.mp4', basename))

    if os.path.exists(mp4_path): 
       return mp4_path
    else: # Deprecated naming convention for idm videos
        return path.join(dirname, re.sub(r'^keys_([a-f0-9]+)\.json$', r'output_\1.mp4', basename))

def has_s3():
    assert find_s3_cfg()

def find_s3_cfg():
    if path.exists(".s3cfg"):
        return ".s3cfg"
    elif path.exists(path.expanduser("~/.s3cfg")):
        return path.expanduser("~/.s3cfg")
    else:
        return None        


def init_boto3_client(config_path = None):
    import boto3
    import configparser

    path = config_path if config_path else find_s3_cfg()
    config = configparser.ConfigParser()
    config.read(path)
    return boto3.client(
        "s3",
        aws_access_key_id=config["default"]["access_key"],
        aws_secret_access_key=config["default"]["secret_key"],
        endpoint_url=f"https://{config['default']['host_base']}",
    )

def upload_to_s3(local_filepath, upload_path, bucket_name, s3client):
    print(f"Uploading {local_filepath} => s3://{bucket_name}/{upload_path}")
    s3client.upload_file(local_filepath, bucket_name, upload_path)

import os
import boto3

def resumable_download_s3_folder(bucket_name: str, s3_folder: str, local_dir: str, s3=None, check_size: bool = True):
    """
    Download the contents of an S3 folder to a local directory, skipping files that already exist.

    :param bucket_name: Name of the S3 bucket.
    :param s3_folder: Folder path in S3 (prefix).
    :param local_dir: Local directory to download to.
    :param s3: path to config for init_boto3_client.
    :param check_size: If True, skip only if local file size matches remote file size.
    """
    s3 = init_boto3_client(config_path=s3)
    files_downloaded = 0
    files_skipped = 0

    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket_name, Prefix=s3_folder):
        if "Contents" not in page:
            continue
        for obj in page["Contents"]:
            key = obj["Key"]
            if key.endswith("/"):
                continue  # Skip directories

            rel_path = os.path.relpath(key, s3_folder)
            local_path = os.path.join(local_dir, rel_path)
            os.makedirs(os.path.dirname(local_path), exist_ok=True)

            # Skip file if it already exists (and optionally matches size)
            if os.path.exists(local_path):
                if not check_size or os.path.getsize(local_path) == obj["Size"]:
                    print(f"Skipping (already exists): {local_path}")
                    files_skipped += 1
                    continue

            print(f"Downloading s3://{bucket_name}/{key} => {local_path}")
            s3.download_file(bucket_name, key, local_path)
            files_downloaded += 1

    assert files_downloaded + files_skipped > 0, f"No files were found in s3://{bucket_name}/{s3_folder}"
    print(f"Downloaded: {files_downloaded}, Skipped: {files_skipped}")


def download_s3_folder(bucket_name: str, s3_folder: str, local_dir: str, s3=None):
    """
    Download the contents of a folder directory from S3 to a local path.
    
    :param bucket_name: Name of the S3 bucket.
    :param s3_folder: Folder path in S3 (prefix).
    :param local_dir: Local directory to download to.
    """
    s3 = init_boto3_client(config_path=s3)
    files_downloaded = 0

    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket_name, Prefix=s3_folder):
        if "Contents" not in page:
            continue
        for obj in page["Contents"]:
            key = obj["Key"]
            # Skip "directories"
            if key.endswith("/"):
                continue

            rel_path = os.path.relpath(key, s3_folder)
            local_path = os.path.join(local_dir, rel_path)
            os.makedirs(os.path.dirname(local_path), exist_ok=True)

            print(f"Downloading s3://{bucket_name}/{key} => {local_path}")
            s3.download_file(bucket_name, key, local_path)
            files_downloaded += 1

    assert files_downloaded > 0, "No files were found in s3"

def load_json(path):
    try:
        with open(path, "rb") as f:
            data = orjson.loads(f.read())
    except Exception as e:
        print(f"[JSON ERROR] failed to load {path}")
        raise e

    return data

def save_json(path: str, data) -> None:
    with open(path, "wb") as f:
        f.write(orjson.dumps(data, option=orjson.OPT_INDENT_2))

def train_val_split(dataset, split: float = 0.05):
    from torch.utils.data import Subset
    
    num_samples = len(dataset)
    indices = list(range(num_samples))
    eval_idx = random.sample(indices, round(num_samples*split))
    train_idx = [i for i in indices if i not in eval_idx]

    train_ds, eval_ds = Subset(dataset=dataset, indices=train_idx), Subset(dataset=dataset, indices=eval_idx)
    print(f"[SPLIT] Train size: {len(train_ds)}, Eval size: {len(eval_ds)} of {type(dataset)}")

    return train_ds, eval_ds
