#Download and process NEXRAD radar data

import nexradaws
from pyart import io  # Or wradlib if you switched

class NEXRADCollector:
    def __init__(self):
        self.nexrad = nexradaws.NexradAwsInterface()

    def get_scans(self, station, start_time, end_time):
        return self.nexrad.get_avail_scans(station, start_time, end_time)

    def download_scan(self, scan, dest_folder):
        scan.download_files(dest_folder)

    def process_scan(self, filename):
        radar = io.read_nexrad_archive(filename)
        # Your processing logic here
        return radar
