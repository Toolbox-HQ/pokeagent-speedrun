import os
import orjson
import re 
import os.path as path
from typing import Dict, List

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
    files = list(filter(lambda x: x.endswith(ext), os.listdir(dir)))
    return [os.path.join(dir, file) for file in files]

def map_json_to_mp4(filename):
    dirname, basename = os.path.split(filename)
    new_basename = re.sub(r'^keys_([a-f0-9]+)\.json$', r'output_\1.mp4', basename)
    return path.join(dirname, new_basename)

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

    assert not files_downloaded, "No files were found in s3"

def load_json(path):
    with open(path, "rb") as f:
        data = orjson.loads(f.read())
    return data

def save_json(path: str, data) -> None:
    with open(path, "wb") as f:
        f.write(orjson.dumps(data, option=orjson.OPT_INDENT_2))