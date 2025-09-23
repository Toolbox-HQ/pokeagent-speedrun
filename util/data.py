def init_boto3_client():
    import boto3
    from pathlib import Path
    import configparser

    config = configparser.ConfigParser()
    config.read(Path.joinpath("./.s3cfg"))
    return boto3.client(
        "s3",
        aws_access_key_id=config["default"]["access_key"],
        aws_secret_access_key=config["default"]["secret_key"],
        endpoint_url=f"https://{config['default']['host_base']}",
    )

def upload_to_s3(local_filepath, upload_path, bucket_name, s3client):
    print(f"Uploading {local_filepath} => s3://{bucket_name}{upload_path}")
    s3client.upload_file(local_filepath, bucket_name, upload_path)