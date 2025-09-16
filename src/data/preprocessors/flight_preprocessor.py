import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
import xarray as xr
from scipy import signal
from typing import Dict, List, Tuple, Optional
import warnings

class FlightDataPreprocessor:
    """
    Comprehensive preprocessor for multimodal flight data
    Handles ADS-B, GOES, NEXRAD, and PIREP data integration
    """
    
    def _init_(self, 
                 sequence_length: int = 100,
                 sampling_rate: float = 1.0,
                 normalization_method: str = 'standard',
                 handle_missing: str = 'interpolate'):
        """
        Initialize preprocessor
        
        Args:
            sequence_length: Length of time series sequences
            sampling_rate: Target sampling rate in Hz
            normalization_method: 'standard', 'minmax', or 'robust'
            handle_missing: 'interpolate', 'forward_fill', or 'drop'
        """
        self.sequence_length = sequence_length
        self.sampling_rate = sampling_rate
        self.normalization_method = normalization_method
        self.handle_missing = handle_missing
        
        # Initialize scalers
        self.scalers = {}
        self.feature_names = []
        
        # Define sensor channel mappings
        self.adsb_channels = [
            'altitude', 'ground_speed', 'track', 'vertical_rate',
            'latitude', 'longitude', 'heading'
        ]
        
        self.derived_channels = [
            'acceleration_x', 'acceleration_y', 'acceleration_z',
            'angular_velocity_x', 'angular_velocity_y', 'angular_velocity_z',
            'airspeed_variation', 'altitude_rate_variation'
        ]
        
        self.weather_channels = [
            'temperature', 'pressure', 'humidity', 'wind_speed',
            'wind_direction', 'turbulence_intensity'
        ]
    
    def fit(self, flight_data_dict: Dict) -> 'FlightDataPreprocessor':
        """
        Fit preprocessor on training data
        
        Args:
            flight_data_dict: Dictionary containing different data sources
                - 'adsb': ADS-B flight data
                - 'goes': GOES satellite data  
                - 'nexrad': NEXRAD radar data
                - 'pirep': PIREP reports
        """
        # Process each data source
        processed_data = {}
        
        if 'adsb' in flight_data_dict:
            processed_data['adsb'] = self._process_adsb_data(flight_data_dict['adsb'])
        
        if 'goes' in flight_data_dict:
            processed_data['goes'] = self._process_goes_data(flight_data_dict['goes'])
        
        if 'nexrad' in flight_data_dict:
            processed_data['nexrad'] = self._process_nexrad_data(flight_data_dict['nexrad'])
        
        if 'pirep' in flight_data_dict:
            processed_data['pirep'] = self._process_pirep_data(flight_data_dict['pirep'])
        
        # Combine all processed data
        combined_data = self._combine_multimodal_data(processed_data)
        
        # Fit scalers
        self._fit_scalers(combined_data)
        
        return self
    
    def transform(self, flight_data_dict: Dict) -> np.ndarray:
        """
        Transform flight data using fitted preprocessor
        
        Returns:
            Processed numpy array ready for model input
        """
        # Process each data source
        processed_data = {}
        
        if 'adsb' in flight_data_dict:
            processed_data['adsb'] = self._process_adsb_data(flight_data_dict['adsb'])
        
        if 'goes' in flight_data_dict:
            processed_data['goes'] = self._process_goes_data(flight_data_dict['goes'])
        
        if 'nexrad' in flight_data_dict:
            processed_data['nexrad'] = self._process_nexrad_data(flight_data_dict['nexrad'])
        
        if 'pirep' in flight_data_dict:
            processed_data['pirep'] = self._process_pirep_data(flight_data_dict['pirep'])
        
        # Combine all processed data
        combined_data = self._combine_multimodal_data(processed_data)
        
        # Apply scaling
        scaled_data = self._apply_scaling(combined_data)
        
        # Create sequences
        sequences = self._create_sequences(scaled_data)
        
        return sequences
    
    def fit_transform(self, flight_data_dict: Dict) -> np.ndarray:
        """Fit and transform in one step"""
        return self.fit(flight_data_dict).transform(flight_data_dict)
    
    def _process_adsb_data(self, adsb_df: pd.DataFrame) -> pd.DataFrame:
        """Process ADS-B flight data"""
        df = adsb_df.copy()
        
        # Ensure timestamp is datetime
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp')
        
        # Handle missing values
        df = self._handle_missing_values(df)
        
        # Calculate derived features
        df = self._calculate_derived_features(df)
        
        # Resample to target sampling rate
        if 'timestamp' in df.columns:
            df = self._resample_data(df, self.sampling_rate)
        
        # Filter and smooth data
        df = self._apply_filtering(df)
        
        return df
    
    def _process_goes_data(self, goes_data: List) -> pd.DataFrame:
        """Process GOES satellite data"""
        processed_patches = []
        
        for data_item in goes_data:
            if isinstance(data_item, xr.Dataset):
                # Extract relevant channels (IR, Water Vapor)
                ir_channel = data_item.sel(band=13)  # 10.3 μm IR channel
                wv_channel = data_item.sel(band=8)   # 6.2 μm Water Vapor
                
                # Calculate turbulence indicators
                ir_gradient = self._calculate_spatial_gradient(ir_channel.values)
                wv_gradient = self._calculate_spatial_gradient(wv_channel.values)
                
                # Create features
                features = {
                    'ir_mean': np.mean(ir_channel.values),
                    'ir_std': np.std(ir_channel.values),
                    'ir_gradient_magnitude': np.mean(ir_gradient),
                    'wv_mean': np.mean(wv_channel.values),
                    'wv_std': np.std(wv_channel.values),
                    'wv_gradient_magnitude': np.mean(wv_gradient),
                    'timestamp': data_item.attrs.get('time', pd.Timestamp.now())
                }
                
                processed_patches.append(features)
        
        return pd.DataFrame(processed_patches)
    
    def _process_nexrad_data(self, nexrad_data: List) -> pd.DataFrame:
        """Process NEXRAD radar data"""
        processed_scans = []
        
        for scan_data in nexrad_data:
            # Calculate Eddy Dissipation Rate (EDR) and other turbulence metrics
            if 'spectrum_width' in scan_data and 'reflectivity' in scan_data:
                spectrum_width = scan_data['spectrum_width']
                reflectivity = scan_data['reflectivity']
                
                # Calculate EDR using NEXRAD Turbulence Detection Algorithm
                edr = self._calculate_edr(spectrum_width, reflectivity)
                
                features = {
                    'edr_mean': np.nanmean(edr),
                    'edr_max': np.nanmax(edr),
                    'edr_std': np.nanstd(edr),
                    'reflectivity_mean': np.nanmean(reflectivity),
                    'reflectivity_max': np.nanmax(reflectivity),
                    'spectrum_width_mean': np.nanmean(spectrum_width),
                    'spectrum_width_std': np.nanstd(spectrum_width),
                    'timestamp': scan_data.get('timestamp', pd.Timestamp.now())
                }
                
                processed_scans.append(features)
        
        return pd.DataFrame(processed_scans)
    
    def _process_pirep_data(self, pirep_df: pd.DataFrame) -> pd.DataFrame:
        """Process PIREP pilot reports"""
        df = pirep_df.copy()
        
        # Extract turbulence intensity levels
        if 'turbulence_intensity' in df.columns:
            # Map text descriptions to numeric values
            intensity_map = {
                'NEG': 0, 'SMOOTH': 0, 'LGT': 1, 'MOD': 2, 
                'SEV': 3, 'EXTRM': 4
            }
            df['turbulence_numeric'] = df['turbulence_intensity'].map(intensity_map)
        
        # Create binary turbulence indicator
        df['turbulence_present'] = (df.get('turbulence_numeric', 0) > 0).astype(int)
        
        return df
    
    def _calculate_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate derived features from basic flight parameters"""
        # Calculate accelerations
        if 'ground_speed' in df.columns:
            df['acceleration_x'] = df['ground_speed'].diff() / df['timestamp'].diff().dt.total_seconds()
        
        if 'vertical_rate' in df.columns:
            df['acceleration_z'] = df['vertical_rate'].diff() / df['timestamp'].diff().dt.total_seconds()
        
        # Calculate angular velocities
        if 'track' in df.columns:
            df['angular_velocity_z'] = df['track'].diff() / df['timestamp'].diff().dt.total_seconds()
        
        # Calculate variation measures (turbulence indicators)
        window_size = min(10, len(df) // 4)
        if window_size > 1:
            if 'ground_speed' in df.columns:
                df['airspeed_variation'] = df['ground_speed'].rolling(window=window_size).std()
            
            if 'vertical_rate' in df.columns:
                df['altitude_rate_variation'] = df['vertical_rate'].rolling(window=window_size).std()
        
        return df
    
    def _calculate_spatial_gradient(self, image: np.ndarray) -> np.ndarray:
        """Calculate spatial gradient magnitude for satellite imagery"""
        # Calculate gradients in x and y directions
        grad_x = np.gradient(image, axis=1)
        grad_y = np.gradient(image, axis=0)
        
        # Calculate gradient magnitude
        gradient_magnitude = np.sqrt(grad_x*2 + grad_y*2)
        
        return gradient_magnitude
    
    def _calculate_edr(self, spectrum_width: np.ndarray, reflectivity: np.ndarray) -> np.ndarray:
        """Calculate Eddy Dissipation Rate from radar data"""
        # NEXRAD Turbulence Detection Algorithm (NTDA) implementation
        # EDR ∝ (spectrum_width)^(3/2) for areas with sufficient reflectivity
        
        # Apply reflectivity threshold (typically 5 dBZ)
        valid_mask = reflectivity > 5
        
        # Calculate EDR
        edr = np.zeros_like(spectrum_width)
        edr[valid_mask] = 0.0001 * (spectrum_width[valid_mask] ** 1.5)
        
        # Cap maximum EDR values
        edr = np.clip(edr, 0, 1.0)
        
        return edr
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the data"""
        if self.handle_missing == 'interpolate':
            # Linear interpolation for numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df[numeric_cols] = df[numeric_cols].interpolate(method='linear')
        
        elif self.handle_missing == 'forward_fill':
            df = df.fillna(method='ffill')
        
        elif self.handle_missing == 'drop':
            df = df.dropna()
        
        return df
    
    def _resample_data(self, df: pd.DataFrame, target_rate: float) -> pd.DataFrame:
        """Resample data to target sampling rate"""
        if 'timestamp' not in df.columns:
            return df
        
        df = df.set_index('timestamp')
        
        # Calculate target frequency
        freq = f'{1/target_rate:.0f}S'  # Convert to seconds
        
        # Resample using mean aggregation
        df_resampled = df.resample(freq).mean()
        
        # Reset index
        df_resampled = df_resampled.reset_index()
        
        return df_resampled
    
    def _apply_filtering(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply filtering to smooth noisy sensor data"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if col != 'timestamp':
                # Apply low-pass filter to remove high-frequency noise
                data = df[col].dropna()
                if len(data) > 10:
                    # Butterworth filter
                    sos = signal.butter(2, 0.1, output='sos')
                    filtered_data = signal.sosfiltfilt(sos, data)
                    df.loc[data.index, col] = filtered_data
        
        return df
    
    def _combine_multimodal_data(self, processed_data: Dict) -> pd.DataFrame:
        """Combine data from multiple sources"""
        combined_df = pd.DataFrame()
        
        # Start with ADS-B data as base
        if 'adsb' in processed_data:
            combined_df = processed_data['adsb'].copy()
        
        # Add other data sources through temporal alignment
        for source_name, source_df in processed_data.items():
            if source_name != 'adsb' and not source_df.empty:
                # Align temporally if timestamp columns exist
                if 'timestamp' in combined_df.columns and 'timestamp' in source_df.columns:
                    # Use merge_asof for temporal alignment
                    source_df = source_df.sort_values('timestamp')
                    combined_df = combined_df.sort_values('timestamp')
                    
                    combined_df = pd.merge_asof(
                        combined_df, source_df,
                        on='timestamp',
                        direction='nearest',
                        suffixes=('', f'_{source_name}')
                    )
                else:
                    # Simple concatenation if no timestamps
                    for col in source_df.columns:
                        if col not in combined_df.columns:
                            combined_df[f'{col}_{source_name}'] = source_df[col]
        
        return combined_df
    
    def _fit_scalers(self, data: pd.DataFrame):
        """Fit scalers on the training data"""
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        self.feature_names = list(numeric_cols)
        
        for col in numeric_cols:
            if self.normalization_method == 'standard':
                scaler = StandardScaler()
            elif self.normalization_method == 'minmax':
                scaler = MinMaxScaler()
            else:  # robust scaling
                from sklearn.preprocessing import RobustScaler
                scaler = RobustScaler()
            
            # Fit scaler on non-null values
            valid_data = data[col].dropna().values.reshape(-1, 1)
            if len(valid_data) > 0:
                scaler.fit(valid_data)
                self.scalers[col] = scaler
    
    def _apply_scaling(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply fitted scalers to the data"""
        scaled_data = data.copy()
        
        for col, scaler in self.scalers.items():
            if col in scaled_data.columns:
                valid_mask = ~scaled_data[col].isna()
                if valid_mask.any():
                    scaled_data.loc[valid_mask, col] = scaler.transform(
                        scaled_data.loc[valid_mask, col].values.reshape(-1, 1)
                    ).flatten()
        
        return scaled_data
    
    def _create_sequences(self, data: pd.DataFrame) -> np.ndarray:
        """Create overlapping sequences for time series modeling"""
        # Select only numeric features
        feature_data = data.select_dtypes(include=[np.number])
        
        # Fill any remaining NaN values
        feature_data = feature_data.fillna(0)
        
        # Create sequences
        sequences = []
        for i in range(len(feature_data) - self.sequence_length + 1):
            seq = feature_data.iloc[i:i + self.sequence_length].values
            sequences.append(seq)
        
        return np.array(sequences, dtype=np.float32)
    
    def get_feature_names(self) -> List[str]:
        """Get names of processed features"""
        return self.feature_names.copy()
    
    def save_scalers(self, filepath: str):
        """Save fitted scalers to file"""
        import joblib
        scaler_data = {
            'scalers': self.scalers,
            'feature_names': self.feature_names,
            'config': {
                'sequence_length': self.sequence_length,
                'sampling_rate': self.sampling_rate,
                'normalization_method': self.normalization_method,
                'handle_missing': self.handle_missing
            }
        }
        joblib.dump(scaler_data, filepath)
    
    def load_scalers(self, filepath: str):
        """Load fitted scalers from file"""
        import joblib
        scaler_data = joblib.load(filepath)
        
        self.scalers = scaler_data['scalers']
        self.feature_names = scaler_data['feature_names']
        
        # Load configuration
        config = scaler_data['config']
        self.sequence_length = config['sequence_length']
        self.sampling_rate = config['sampling_rate']
        self.normalization_method = config['normalization_method']
        self.handle_missing = config['handle_missing']