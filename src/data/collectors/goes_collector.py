#Collect GOES satellite data

import boto3
import xarray as xr
from datetime import datetime

class GOESCollector:
    def __init__(self, region='us-east-1'):
        self.s3 = boto3.client('s3', region_name=region)
        self.bucket = 'noaa-goes16'  # Change for goes17 or goes18
    
    def list_objects(self, prefix):
        response = self.s3.list_objects_v2(Bucket=self.bucket, Prefix=prefix)
        return [obj['Key'] for obj in response.get('Contents', [])]
    
    def download_file(self, key, dest_path):
        self.s3.download_file(self.bucket, key, dest_path)
    
    def get_satellite_data(self, product, date, hour, channel=None):
        # date: datetime object
        prefix = f"{product}/{date:%Y/%j}/{hour:02d}/"
        keys = self.list_objects(prefix)
        if channel:
            keys = [k for k in keys if f"C{channel:02d}" in k]
        for key in keys:
            dest = f"./data/raw/goes/{key.split('/')[-1]}"
            self.download_file(key, dest)
            # Load with xarray if needed:
            # ds = xr.open_dataset(dest)
