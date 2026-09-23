import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler, LabelEncoder

class SignalDataset(Dataset):
    def __init__(self, ecg_signals, ppg_signals, features, target_sbp, target_dbp, 
                 signal_scaler=None, feature_scaler=None, feature_columns=None, 
                 target_stats=None, max_length=2500, input_signals='ppg_ecg'):
        self.ecg_signals = ecg_signals
        self.ppg_signals = ppg_signals
        self.features = features
        self.target_stats = target_stats or self._calculate_target_stats(target_sbp, target_dbp)
        self.target_sbp = self._standardize_target(target_sbp, 'sbp')
        self.target_dbp = self._standardize_target(target_dbp, 'dbp')
        self.max_length = max_length
        self.feature_columns = feature_columns
        self.input_signals = input_signals
        
        # Normalize signals
        if signal_scaler is None:
            self.signal_scaler = StandardScaler()
            # Fit on flattened signals
            all_signals = []
            for i in range(len(ecg_signals)):
                if input_signals == 'ppg_ecg':
                    ecg_truncated = ecg_signals[i][:max_length] if len(ecg_signals[i]) > max_length else ecg_signals[i]
                    ppg_truncated = ppg_signals[i][:max_length] if len(ppg_signals[i]) > max_length else ppg_signals[i]
                    combined = np.concatenate([ecg_truncated, ppg_truncated])
                elif input_signals == 'ppg':
                    ppg_truncated = ppg_signals[i][:max_length] if len(ppg_signals[i]) > max_length else ppg_signals[i]
                    combined = ppg_truncated
                all_signals.extend(combined)
            
            self.signal_scaler.fit(np.array(all_signals).reshape(-1, 1))
        else:
            self.signal_scaler = signal_scaler
        
        # Process features if provided
        self.feature_scaler = feature_scaler
        if feature_columns is not None and len(feature_columns) > 0:
            self.processed_features = self._process_features(features[feature_columns])
        else:
            self.processed_features = None

    @staticmethod
    def _calculate_target_stats(target_sbp, target_dbp):
        sbp = np.asarray(target_sbp, dtype=np.float64)
        dbp = np.asarray(target_dbp, dtype=np.float64)
        sbp_std = float(np.std(sbp))
        dbp_std = float(np.std(dbp))

        if sbp_std == 0 or dbp_std == 0:
            raise ValueError(
                f"Cannot standardize constant BP targets: SBP std={sbp_std}, DBP std={dbp_std}"
            )

        return {
            'sbp_mean': float(np.mean(sbp)),
            'sbp_std': sbp_std,
            'dbp_mean': float(np.mean(dbp)),
            'dbp_std': dbp_std
        }

    def _standardize_target(self, target, target_name):
        target = np.asarray(target, dtype=np.float32)
        mean = self.target_stats[f'{target_name}_mean']
        std = self.target_stats[f'{target_name}_std']
        return (target - mean) / std
    
    def _process_features(self, features_df):
        processed = features_df.copy()
        
        # Handle categorical variables
        for col in processed.columns:
            if processed[col].dtype == 'object':
                le = LabelEncoder()
                processed[col] = le.fit_transform(processed[col].astype(str))
        
        # Scale features if scaler is provided and fitted
        if self.feature_scaler is not None:
            # Check if scaler is fitted
            try:
                processed = pd.DataFrame(
                    self.feature_scaler.transform(processed),
                    columns=processed.columns
                )
            except:
                # If not fitted, fit and transform
                processed = pd.DataFrame(
                    self.feature_scaler.fit_transform(processed),
                    columns=processed.columns
                )
        
        return processed.values.astype(np.float32)
    
    def _pad_or_truncate(self, signal, max_length):
        if len(signal) > max_length:
            return signal[:max_length]
        elif len(signal) < max_length:
            padding = np.zeros(max_length - len(signal))
            return np.concatenate([signal, padding])
        return signal
    
    def __len__(self):
        return len(self.ecg_signals)
    
    def __getitem__(self, idx):
        # Process PPG signal (always included)
        ppg = self._pad_or_truncate(self.ppg_signals[idx], self.max_length)
        ppg_norm = self.signal_scaler.transform(ppg.reshape(-1, 1)).flatten()
        
        if self.input_signals == 'ppg_ecg':
            # Process ECG signal
            ecg = self._pad_or_truncate(self.ecg_signals[idx], self.max_length)
            ecg_norm = self.signal_scaler.transform(ecg.reshape(-1, 1)).flatten()
            # Combine signals (ECG and PPG as two channels)
            signals = np.stack([ecg_norm, ppg_norm], axis=0)  # Shape: (2, max_length)
        elif self.input_signals == 'ppg':
            # Only PPG signal
            signals = ppg_norm.reshape(1, -1)  # Shape: (1, max_length)
        
        # Get targets
        sbp = self.target_sbp[idx]
        dbp = self.target_dbp[idx]
        
        result = {
            'signals': torch.FloatTensor(signals),
            'sbp': torch.FloatTensor([sbp]),
            'dbp': torch.FloatTensor([dbp])
        }
        
        # Add features if available
        if self.processed_features is not None:
            result['features'] = torch.FloatTensor(self.processed_features[idx])
        
        return result

def _prepare_feature_columns(features_df):
    """Add compatible aliases without removing columns used by result export."""
    features_df = features_df.copy()
    if 'height' not in features_df.columns and 'hight' in features_df.columns:
        features_df['height'] = features_df['hight']
    return features_df


def _calculate_bp(features_df):
    """Return the BP target definition shared by both data-loading protocols."""
    # sbp = (features_df['sbppre'] + features_df['sbppost']) / 2
    # dbp = (features_df['dbppre'] + features_df['dbppost']) / 2
    sbp = features_df['sbppre']
    dbp = features_df['dbppre']
    return sbp.values, dbp.values


def _load_measurements(data_dir, measurement_numbers):
    """Load and concatenate one or more measurement sessions in input order."""
    if not measurement_numbers:
        raise ValueError("At least one measurement number must be provided")
    if len(set(measurement_numbers)) != len(measurement_numbers):
        raise ValueError(
            f"Duplicate measurement numbers are not allowed: {measurement_numbers}"
        )

    all_ecg = []
    all_ppg = []
    all_features = []

    for measurement_number in measurement_numbers:
        signals_path = os.path.join(
            data_dir, f'measurement_{measurement_number}_signals_processed.npy'
        )
        features_path = os.path.join(
            data_dir, f'measurement_{measurement_number}_features.csv'
        )

        if not os.path.exists(signals_path) or not os.path.exists(features_path):
            raise FileNotFoundError(
                f"Missing files for measurement {measurement_number}: "
                f"{signals_path} and/or {features_path}"
            )

        signals = np.load(signals_path, allow_pickle=True).item()
        features = _prepare_feature_columns(pd.read_csv(features_path))
        ecg_signals = signals['ecg_signals']
        ppg_signals = signals['ppg_signals']

        if not (len(ecg_signals) == len(ppg_signals) == len(features)):
            raise ValueError(
                f"Measurement {measurement_number} row mismatch: "
                f"ECG={len(ecg_signals)}, PPG={len(ppg_signals)}, "
                f"features={len(features)}"
            )

        all_ecg.extend(ecg_signals)
        all_ppg.extend(ppg_signals)
        all_features.append(features)
        print(
            f"Loaded measurement {measurement_number}: "
            f"{len(features)} samples, {features['subject_id'].nunique()} subjects"
        )

    return all_ecg, all_ppg, pd.concat(all_features, ignore_index=True)


def prepare_data(data_dir, fold_number=None, data_format='subject_cv',
                 train_measurements=None, test_measurements=None):
    """Load subject-CV folds or measurement-based train/test sessions."""
    if data_format == 'subject_cv':
        if fold_number is None:
            raise ValueError("fold_number is required when data_format='subject_cv'")

        fold_dir = os.path.join(data_dir, f'fold_{fold_number}')
        train_signals = np.load(
            os.path.join(fold_dir, 'train_signals.npy'), allow_pickle=True
        ).item()
        train_features = _prepare_feature_columns(
            pd.read_csv(os.path.join(fold_dir, 'train_features.csv'))
        )
        test_signals = np.load(
            os.path.join(fold_dir, 'test_signals.npy'), allow_pickle=True
        ).item()
        test_features = _prepare_feature_columns(
            pd.read_csv(os.path.join(fold_dir, 'test_features.csv'))
        )
        train_ecg = train_signals['ecg_signals']
        train_ppg = train_signals['ppg_signals']
        test_ecg = test_signals['ecg_signals']
        test_ppg = test_signals['ppg_signals']
    elif data_format == 'measurements':
        train_measurements = train_measurements or [1, 2, 3]
        test_measurements = test_measurements or [4]
        overlap = set(train_measurements) & set(test_measurements)
        if overlap:
            raise ValueError(
                f"Train and test measurement lists overlap: {sorted(overlap)}"
            )

        print(
            f"Measurement protocol: train={train_measurements}, "
            f"test={test_measurements}"
        )
        train_ecg, train_ppg, train_features = _load_measurements(
            data_dir, train_measurements
        )
        test_ecg, test_ppg, test_features = _load_measurements(
            data_dir, test_measurements
        )

        train_subjects = set(train_features['subject_id'].astype(str))
        test_subjects = set(test_features['subject_id'].astype(str))
        shared_subjects = train_subjects & test_subjects
        print(
            "Measurement protocol subject overlap: "
            f"{len(shared_subjects)}/{len(test_subjects)} test subjects "
            "also occur in training"
        )
        print(
            f"Combined measurement samples: train={len(train_features)}, "
            f"test={len(test_features)}"
        )
    else:
        raise ValueError(
            "data_format must be either 'subject_cv' or 'measurements'"
        )

    train_sbp, train_dbp = _calculate_bp(train_features)
    test_sbp, test_dbp = _calculate_bp(test_features)

    return {
        'train_ecg': train_ecg,
        'train_ppg': train_ppg,
        'train_features': train_features,
        'train_sbp': train_sbp,
        'train_dbp': train_dbp,
        'test_ecg': test_ecg,
        'test_ppg': test_ppg,
        'test_features': test_features,
        'test_sbp': test_sbp,
        'test_dbp': test_dbp
    }

def prepare_feature_scaler(features_df, feature_columns):
    """Prepare and fit feature scaler"""
    if not feature_columns:
        return None
    
    # Process categorical variables
    processed_features = features_df[feature_columns].copy()
    for col in processed_features.columns:
        if processed_features[col].dtype == 'object':
            le = LabelEncoder()
            processed_features[col] = le.fit_transform(processed_features[col].astype(str))
    
    # Fit scaler
    scaler = StandardScaler()
    scaler.fit(processed_features)
    return scaler
