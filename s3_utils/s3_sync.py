import boto3
from botocore.exceptions import ClientError
import configparser
import os
import argparse
from tqdm import tqdm

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
    prefix = prefix.lstrip("/")

    if prefix != ".cache":
        print(f"[WARNING]: Non-standard downdload location")

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

def s3_exists(bucket, key, s3):
    try:
        s3.head_object(Bucket=bucket, Key=key)
        return True
    except ClientError as e:
        if e.response["Error"]["Code"] == "404":
            return False
        raise

def upload_to_s3(local_filepath, upload_path, bucket_name, s3client):
    s3client.upload_file(local_filepath, bucket_name, upload_path)

def sync_folder(local: str, remote: str, bucket: str, dry_run: bool = True):
    
    s3 = init_boto3_client()

    local_paths = [os.path.join(dp, f) for dp, dn, filenames in os.walk(local) for f in filenames]
    keys = [p.replace(local.rstrip('/'), remote.rstrip('/'), 1) for p in local_paths]    
    upload_paths = [f"s3://{bucket}/{p}" for p in keys]
    pbar = tqdm(list(zip(local_paths, keys, upload_paths)))
    
    for l, k, r in pbar:
        if dry_run: 
            print(f"[UPLOAD]: {l} => {r}")
        else:
            if s3_exists(bucket, k, s3):
                print(f"[SKIP]: {l} => {r}")
            else:
                s3.upload_file(l, bucket, k)
                print(f"[UPLOADED]: {l} => {r}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sync a local folder to an S3 bucket prefix.")

    parser.add_argument(
        "local",
        type=str,
        help="Local folder to sync from",
    )
    parser.add_argument(
        "remote",
        type=str,
        help="Remote S3 prefix to sync to",
    )

    parser.add_argument(
        "--bucket",
        type=str,
        required=True,
        help="S3 bucket name",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print operations without uploading",
    )

    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        help="Upload or download the directory",
    )

    args = parser.parse_args()

    if args.sync == "upload":
        sync_folder(
            local=args.local,
            remote=args.remote,
            bucket=args.bucket,
            dry_run=args.dry_run,
        )
    elif args.sync == "sync":
        download_prefix(args.bucket, args.remote, args.local)
    else:
        raise Exception("MalformedCommandError")