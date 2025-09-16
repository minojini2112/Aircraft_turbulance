import torch
from torch.utils.data import Dataset
import numpy as np
import xarray as xr
import pandas as pd

class FlightSensorDataset(Dataset):
    def __init__(self, data_array, seq_length=100, transform=None):
        """
        Dataset for multivariate flight sensor time series.

        Args:
        - data_array: numpy array or torch tensor (num_timesteps, num_channels)
        - seq_length: time window length for each sample
        - transform: torch transform or callable to apply on each sample
        """
        self.data = data_array
        self.seq_length = seq_length
        self.transform = transform

    def __len__(self):
        return len(self.data) - self.seq_length + 1

    def __getitem__(self, idx):
        sample = self.data[idx : idx + self.seq_length]
        if self.transform:
            sample = self.transform(sample)
        sample = torch.tensor(sample, dtype=torch.float32)
        return sample


class GOESSatelliteDataset(Dataset):
    def __init__(self, file_paths, patch_size=(32, 32), transform=None):
        """
        Dataset for GOES satellite image patches.

        Args:
        - file_paths: list of GOES NetCDF file paths
        - patch_size: size of square patches to extract from images
        - transform: optional transform to apply on patches
        """
        self.files = file_paths
        self.patch_size = patch_size
        self.transform = transform
        self.data = []
        self._load_patches()

    def _load_patches(self):
        for f in self.files:
            ds = xr.open_dataset(f)
            # Assuming variable 'CMI' contains imagery
            img = ds['CMI'].values
            h, w = img.shape[-2], img.shape[-1]
            for i in range(0, h, self.patch_size[0]):
                for j in range(0, w, self.patch_size[1]):
                    patch = img[..., i:i+self.patch_size[0], j:j+self.patch_size[1]]
                    if patch.shape[-2:] == self.patch_size:
                        self.data.append(patch)
            ds.close()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        patch = self.data[idx]
        if self.transform:
            patch = self.transform(patch)
        patch = torch.tensor(patch, dtype=torch.float32)
        return patch


class RadarScanDataset(Dataset):
    def __init__(self, radar_files, transform=None):
        """
        Dataset to load radar scan data from NEXRAD or other radar archives.

        Args:
        - radar_files: list of radar data filepaths
        - transform: optional transform on radar data slices
        """
        self.files = radar_files
        self.transform = transform
        self.data = []
        self._load_data()

    def _load_data(self):
        for file in self.files:
            # Load with wradlib or pyart here (wradlib recommended)
            import wradlib
            radar_data = wradlib.io.read(file)
            self.data.append(radar_data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        scan = self.data[idx]
        if self.transform:
            scan = self.transform(scan)
        return scan


class PIREPDataset(Dataset):
    def __init__(self, pirep_df, transform=None):
        """
        Dataset for textual Pilot Reports (PIREPs) annotated with turbulence info.

        Args:
        - pirep_df: pandas dataframe with columns like ['timestamp', 'latitude', 'longitude', 'turbulence_intensity']
        - transform: optional transform for text or categorical encoding
        """
        self.pirep_df = pirep_df
        self.transform = transform

    def __len__(self):
        return len(self.pirep_df)

    def __getitem__(self, idx):
        row = self.pirep_df.iloc[idx]
        sample = {
            'timestamp': row['timestamp'],
            'latitude': row['latitude'],
            'longitude': row['longitude'],
            'turbulence_intensity': row['turbulence_intensity']
        }
        if self.transform:
            sample = self.transform(sample)

        return sample
