#Collect ADS-B flight data

from pyopensky import OpenSkyApi
import pandas as pd

class ADSBCollector:
    def __init__(self):
        self.api = OpenSkyApi()

    def collect_flight_data(self, bbox, time_start, time_end):
        states = self.api.get_states(time_secs=time_start, bbox=bbox)
        data = []
        for state in states.states:
            data.append({
                'icao24': state.icao24,
                'callsign': state.callsign,
                'longitude': state.longitude,
                'latitude': state.latitude,
                'altitude': state.geo_altitude,
                'velocity': state.velocity,
                'vertical_rate': state.vertical_rate,
                'heading': state.heading,
                'timestamp': time_start
            })
        return pd.DataFrame(data)
