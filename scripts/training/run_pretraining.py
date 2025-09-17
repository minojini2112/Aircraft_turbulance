#!/usr/bin/env python3
import sys
import os
# Add project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import argparse
import torch
from torch.utils.data import DataLoader, TensorDataset
import yaml

from src.data.preprocessors.flight_preprocessor import FlightDataPreprocessor
from src.models.mae.flight_mae import build_flight_mae_model

def load_raw_data(cfg):
    import pandas as pd
    import numpy as np
    import xarray as xr
    from src.data.collectors.adsb_collector import ADSBCollector
    from src.data.collectors.goes_collector import GOESCollector
    import os
    from datetime import datetime, timedelta

    # Check if data files exist, if not create sample data
    data = {}
    
    # ADS-B data
    adsb_path = cfg['data']['adsb_csv']
    if os.path.exists(adsb_path):
        adsb_df = pd.read_csv(adsb_path)
    else:
        print(f"ADS-B data file not found at {adsb_path}. Creating sample data...")
        # Create sample ADS-B data
        adsb_collector = ADSBCollector()
        # Use a default bounding box (US East Coast) and recent time range
        bbox = (39.0, 42.0, -75.0, -70.0)  # lat_min, lat_max, lon_min, lon_max
        time_end = int(datetime.now().timestamp())
        time_start = time_end - 3600  # 1 hour ago
        
        try:
            adsb_df = adsb_collector.collect_flight_data(bbox, time_start, time_end)
            if adsb_df.empty:
                raise ValueError("No ADS-B data collected")
        except Exception as e:
            print(f"Failed to collect real ADS-B data: {e}. Creating synthetic data...")
            # Create synthetic ADS-B data
            n_samples = 5000
            adsb_df = pd.DataFrame({
                'timestamp': pd.date_range(start=datetime.now() - timedelta(hours=1), periods=n_samples, freq='1min'),
                'altitude': np.random.normal(35000, 5000, n_samples),
                'ground_speed': np.random.normal(500, 100, n_samples),
                'track': np.random.uniform(0, 360, n_samples),
                'vertical_rate': np.random.normal(0, 500, n_samples),
                'latitude': np.random.uniform(39.0, 42.0, n_samples),
                'longitude': np.random.uniform(-75.0, -70.0, n_samples),
                'heading': np.random.uniform(0, 360, n_samples)
            })
    
    data['adsb'] = adsb_df

    # GOES data
    goes_files = cfg['data']['goes_files']
    goes = []
    for f in goes_files:
        if os.path.exists(f):
            goes.append(xr.open_dataset(f))
        else:
            print(f"GOES file not found at {f}. Creating sample data...")
            # Create synthetic GOES data with proper band structure
            synthetic_goes = xr.Dataset({
                'data': (['band', 'y', 'x'], np.random.normal(200, 20, (16, 100, 100)))
            }, coords={
                'band': range(1, 17),  # GOES bands 1-16
                'y': np.linspace(25, 50, 100),
                'x': np.linspace(-80, -65, 100)
            })
            synthetic_goes.attrs['time'] = datetime.now()
            goes.append(synthetic_goes)
    
    data['goes'] = goes
    return data

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, help="Path to pretrain config YAML")
    p.add_argument("--output", default="models/pretrained/encoder.pth",
                   help="Where to save pretrained encoder")
    args = p.parse_args()

    cfg = yaml.safe_load(open(args.config))

    # 1. Preprocess
    pre = FlightDataPreprocessor(
        sequence_length=cfg['preprocessing']['sequence_length'],
        sampling_rate=cfg['preprocessing']['sampling_rate'],
        normalization_method=cfg['preprocessing']['normalization'],
        handle_missing=cfg['preprocessing']['handle_missing']
    )
    raw = load_raw_data(cfg)
    data = pre.fit_transform(raw)  # shape (N, seq_len, C)

    # 2. Dataset & Dataloader
    tensor_data = torch.from_numpy(data)
    ds = TensorDataset(tensor_data)
    loader = DataLoader(ds, batch_size=cfg['training']['batch_size'], shuffle=True)

    # 3. Model setup
    device = torch.device(cfg['training']['device'])
    model = build_flight_mae_model().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['training']['lr'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg['training']['epochs'])

    # 4. Training loop
    model.train()
    for epoch in range(cfg['training']['epochs']):
        total_loss = 0.0
        for (batch,) in loader:
            batch = batch.to(device)
            loss, _, _ = model(batch, mask_ratio=cfg['training']['mask_ratio'])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()
        print(f"[Pretrain] Epoch {epoch+1}/{cfg['training']['epochs']}  Loss: {total_loss/len(loader):.6f}")

    # 5. Save encoder
    torch.save(model.encoder.state_dict(), args.output)
    print(f"Pretrained encoder saved to {args.output}")

if __name__ == "__main__":
    main()