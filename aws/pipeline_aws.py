import os
import sys
import argparse
from pathlib import Path

import boto3


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def upload_to_s3(bucket_name, local_path, s3_key, region='ap-southeast-1'):
    s3 = boto3.client('s3', region_name=region)
    print(f'Uploading {local_path} -> s3://{bucket_name}/{s3_key}')
    s3.upload_file(str(local_path), bucket_name, s3_key)
    print('Upload complete.')


def download_from_s3(bucket_name, s3_key, local_path, region='ap-southeast-1'):
    s3 = boto3.client('s3', region_name=region)
    Path(local_path).parent.mkdir(parents=True, exist_ok=True)
    print(f'Downloading s3://{bucket_name}/{s3_key} -> {local_path}')
    s3.download_file(bucket_name, s3_key, str(local_path))
    print('Download complete.')


def create_bucket_if_needed(bucket_name, region='ap-southeast-1'):
    s3 = boto3.client('s3', region_name=region)
    try:
        s3.head_bucket(Bucket=bucket_name)
        print(f'Bucket already exists: {bucket_name}')
    except Exception:
        if region == 'us-east-1':
            s3.create_bucket(Bucket=bucket_name)
        else:
            s3.create_bucket(
                Bucket=bucket_name,
                CreateBucketConfiguration={'LocationConstraint': region},
            )
        print(f'Bucket created: {bucket_name}')


def upload_project_artifacts(bucket_name, region='ap-southeast-1'):
    create_bucket_if_needed(bucket_name, region)

    files_to_upload = [
        (PROJECT_ROOT / 'data' / 'data_C.csv', 'data/data_C.csv'),
        (PROJECT_ROOT / 'models' / 'best_model.pkl', 'models/best_model.pkl'),
        (PROJECT_ROOT / 'models' / 'preprocessor.pkl', 'models/preprocessor.pkl'),
        (PROJECT_ROOT / 'models' / 'label_encoder.pkl', 'models/label_encoder.pkl'),
        (PROJECT_ROOT / 'models' / 'feature_columns.pkl', 'models/feature_columns.pkl'),
    ]

    for local_path, s3_key in files_to_upload:
        if local_path.exists():
            upload_to_s3(bucket_name, local_path, s3_key, region)
        else:
            print(f'Skipped (not found): {local_path}')


def run_training_on_ec2(bucket_name, region='ap-southeast-1'):
    sys.path.insert(0, str(PROJECT_ROOT / 'pipeline'))
    from main_pipeline import run_pipeline

    data_local = PROJECT_ROOT / 'data' / 'data_C.csv'
    if not data_local.exists():
        download_from_s3(bucket_name, 'data/data_C.csv', data_local, region)

    best_result, results = run_pipeline(
        data_path=data_local,
        models_dir=PROJECT_ROOT / 'models',
        experiment_name='credit_score_aws',
    )

    for filename in ['best_model.pkl', 'preprocessor.pkl',
                     'label_encoder.pkl', 'feature_columns.pkl']:
        upload_to_s3(
            bucket_name,
            PROJECT_ROOT / 'models' / filename,
            f'models/{filename}',
            region,
        )

    return best_result


def parse_args():
    parser = argparse.ArgumentParser(description='AWS pipeline for credit score model')
    parser.add_argument('action', choices=['upload', 'train', 'download'],
                        help='upload: push data+models to S3 | '
                             'train: run pipeline locally and upload to S3 | '
                             'download: pull model from S3')
    parser.add_argument('--bucket', required=True, help='S3 bucket name')
    parser.add_argument('--region', default='ap-southeast-1', help='AWS region')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    if args.action == 'upload':
        upload_project_artifacts(args.bucket, args.region)
    elif args.action == 'train':
        run_training_on_ec2(args.bucket, args.region)
    elif args.action == 'download':
        for filename in ['best_model.pkl', 'preprocessor.pkl',
                         'label_encoder.pkl', 'feature_columns.pkl']:
            download_from_s3(
                args.bucket,
                f'models/{filename}',
                PROJECT_ROOT / 'models' / filename,
                args.region,
            )
