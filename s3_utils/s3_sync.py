import boto3
import configparser
import os

def init_boto3_client():
    config = configparser.ConfigParser()
    config.read(os.path.join(os.path.expanduser("~"), ".s3cfg"))
    return boto3.client(
        "s3",
        aws_access_key_id=config["default"]["access_key"],
        aws_secret_access_key=config["default"]["secret_key"],
        endpoint_url=f"https://{config['default']['host_base']}",
    )

def download_prefix(bucket: str, prefix: str, cache_root: str = ".cache", s3=None):
    """
    Accepts either a folder-like prefix or a full object key.
    Downloads to cache/{key}, skipping files that already exist.
    """
    s3 = s3 or init_boto3_client()
    prefix = prefix.lstrip("/")  # do NOT force a trailing slash

    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith("/"):
                continue
            local_path = os.path.join(cache_root, key)
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            if os.path.exists(local_path):
                continue
            s3.download_file(bucket, key, local_path)

def check_s3_existing_videos(bucket_name: str, s3_prefix: str = "youtube_videos", video_ids: list = None):
    s3client = init_boto3_client()
    try:
        paginator = s3client.get_paginator("list_objects_v2")
        page_iterator = paginator.paginate(Bucket=bucket_name, Prefix=s3_prefix)

        existing_files = set()
        for page in page_iterator:
            if "Contents" not in page:
                continue
            for obj in page["Contents"]:
                key = obj["Key"].split("/")[-1].replace(".mp4", "")
                existing_files.add(key)

        return existing_files.intersection(set(video_ids)) if video_ids else existing_files

    except Exception as e:
        print(f"[ERROR] Error listing S3 objects: {e}")
        return set()

def upload_to_s3(local_filepath, upload_path, bucket_name, s3client):
    s3client.upload_file(local_filepath, bucket_name, upload_path)