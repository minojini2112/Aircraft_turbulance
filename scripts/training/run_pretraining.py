#!/usr/bin/env python3
import argparse
import torch
from torch.utils.data import DataLoader, TensorDataset
import yaml

from src.data.preprocessors.flight_preprocessor import FlightDataPreprocessor
from src.models.mae.flight_mae import build_flight_mae_model

def load_raw_data(cfg):
    import pandas as pd
    import xarray as xr
    from src.data.collectors.adsb_collector import ADSBCollector
    from src.data.collectors.goes_collector import GOESCollector

    # ADS-B
    adsb_df = pd.read_csv(cfg['data']['adsb_csv'])

    # GOES
    goes = []
    for f in cfg['data']['goes_files']:
        goes.append(xr.open_dataset(f))
    return {'adsb': adsb_df, 'goes': goes}

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