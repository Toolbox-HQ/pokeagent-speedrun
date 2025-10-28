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

def download_prefix(bucket: str, prefix: str, cache_root: str = "cache", s3=None):
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