#Collect PIREP pilot reports

import requests
import pandas as pd

class PIREPCollector:
    def __init__(self, base_url="https://mesonet.agron.iastate.edu"):
        self.base_url = base_url
    
    def fetch_pireps(self, start_date, end_date):
        params = {
            "sts": start_date,
            "ets": end_date,
            "phenomena[]": "PIREP",
            "format": "csv"
        }
        url = f"{self.base_url}/request/gis/pireps.py"
        response = requests.get(url, params=params)
        if response.status_code == 200:
            df = pd.read_csv(pd.compat.StringIO(response.text))
            return df
        else:
            raise Exception(f"Failed to fetch PIREP data: {response.status_code}")
