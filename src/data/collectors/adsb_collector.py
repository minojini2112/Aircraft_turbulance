#Collect ADS-B flight data

from pyopensky.rest import REST
import pandas as pd

class ADSBCollector:
    def __init__(self):
        self.api = REST()

    def collect_flight_data(self, bbox, time_start, time_end):
        try:
            # Use the states method to get current aircraft states
            states = self.api.states()
            data = []
            
            if states is not None and hasattr(states, 'states'):
                for state in states.states:
                    # Check if aircraft is within bounding box
                    if (bbox[0] <= state.latitude <= bbox[1] and 
                        bbox[2] <= state.longitude <= bbox[3]):
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
        except Exception as e:
            print(f"Error collecting ADS-B data: {e}")
            return pd.DataFrame()
