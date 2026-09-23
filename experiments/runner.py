import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader

from models.transformer import BPTransformer, count_parameters
from data.dataset import SignalDataset, prepare_data, prepare_feature_scaler
from training.trainer import train_model, evaluate_model_with_details
from utils.metrics import calculate_fairness_metrics

def run_stratified_model(data, stratify_type, run_name, output_dir, batch_size=128, 
                        d_model=16, nhead=4, num_encoder_layers=2, dim_feedforward=64,
                        input_signals='ppg_ecg', num_epochs=100):
    """Run stratified model based on stratification type"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n=== Training {stratify_type} stratified model ===")
    
    # Define grouping strategy
    if stratify_type == 'stratified_hypertension':
        # Group by hypertension history
        def get_groups(df):
            return df.groupby('history of hypertension')
        group_name = 'hypertension'
        
    elif stratify_type == 'stratified_gender':
        # Group by gender
        def get_groups(df):
            return df.groupby('gender')
        group_name = 'gender'
        
    elif stratify_type == 'stratified_hypertension_gender':
        # Group by both hypertension and gender
        def get_groups(df):
            return df.groupby(['history of hypertension', 'gender'])
        group_name = 'hypertension_gender'
    
    # Group data
    train_groups = get_groups(data['train_features'])
    test_groups = get_groups(data['test_features'])
    
    all_predictions = []
    group_results = []
    
    # Get all possible groups
    all_train_groups = list(train_groups.groups.keys())
    all_test_groups = list(test_groups.groups.keys())
    all_groups = list(set(all_train_groups + all_test_groups))
    
    for group_key in all_groups:
        if group_key not in train_groups.groups or group_key not in test_groups.groups:
            print(f"Skipping group {group_key} - not present in both train and test sets")
            continue
            
        print(f"\n--- Training model for {group_name}={group_key} ---")
        
        # Get indices for this group
        train_indices = train_groups.groups[group_key].values
        test_indices = test_groups.groups[group_key].values
        
        # Extract data for this group
        group_train_ecg = [data['train_ecg'][i] for i in train_indices]
        group_train_ppg = [data['train_ppg'][i] for i in train_indices]
        group_train_features = data['train_features'].iloc[train_indices]
        group_train_sbp = data['train_sbp'][train_indices]
        group_train_dbp = data['train_dbp'][train_indices]
        
        group_test_ecg = [data['test_ecg'][i] for i in test_indices]
        group_test_ppg = [data['test_ppg'][i] for i in test_indices]
        group_test_features = data['test_features'].iloc[test_indices]
        group_test_sbp = data['test_sbp'][test_indices]
        group_test_dbp = data['test_dbp'][test_indices]
        
        print(f"Group {group_key}: Train={len(group_train_ecg)}, Test={len(group_test_ecg)}")
        
        if len(group_train_ecg) < 10 or len(group_test_ecg) < 1:
            print(f"Skipping group {group_key} due to insufficient data")
            continue
        
        # Create datasets for this group
        train_dataset = SignalDataset(
            group_train_ecg, group_train_ppg, group_train_features,
            group_train_sbp, group_train_dbp,
            feature_columns=[], input_signals=input_signals
        )
        
        test_dataset = SignalDataset(
            group_test_ecg, group_test_ppg, group_test_features,
            group_test_sbp, group_test_dbp,
            signal_scaler=train_dataset.signal_scaler,
            target_stats=train_dataset.target_stats,
            feature_columns=[], input_signals=input_signals
        )
        
        # Split training data for validation
        val_size = max(1, int(0.2 * len(train_dataset)))
        train_size = len(train_dataset) - val_size
        train_subset, val_subset = torch.utils.data.random_split(train_dataset, [train_size, val_size])
        
        # Create data loaders
        train_loader = DataLoader(train_subset, batch_size=min(batch_size, len(train_subset)), shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=min(batch_size, len(val_subset)), shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=min(batch_size, len(test_dataset)), shuffle=False)
        
        # Create model for this group
        model = BPTransformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            dim_feedforward=dim_feedforward,
            num_features=0,
            input_signals=input_signals
        )
        
        # Train model for this group
        model = train_model(
            model, train_loader, val_loader, train_dataset.target_stats,
            num_epochs=num_epochs, device=device
        )
        
        # Evaluate model on this group
        test_results = evaluate_model_with_details(
            model, test_loader, group_test_features, train_dataset.target_stats, device=device
        )
        
        # Store predictions with group info
        for prediction in test_results['predictions']:
            # Add stratification-specific information
            prediction['group_key'] = str(group_key)
            prediction['model_type'] = stratify_type
            
            if stratify_type == 'stratified_hypertension':
                prediction['hypertension_group'] = group_key
            elif stratify_type == 'stratified_gender':
                prediction['gender_group'] = group_key
            elif stratify_type == 'stratified_hypertension_gender':
                prediction['hypertension_group'] = group_key[0]
                prediction['gender_group'] = group_key[1]
            
            all_predictions.append(prediction)
        
        group_results.append({
            'group': str(group_key),
            'stratify_type': stratify_type,
            'num_params': count_parameters(model),
            'train_size': len(group_train_ecg),
            'test_size': len(group_test_ecg),
            'sbp_mae': test_results['sbp_mae'],
            'sbp_rmse': test_results['sbp_rmse'],
            'sbp_std': test_results['sbp_std'],
            'sbp_me': test_results['sbp_me'],
            'dbp_mae': test_results['dbp_mae'],
            'dbp_rmse': test_results['dbp_rmse'],
            'dbp_std': test_results['dbp_std'],
            'dbp_me': test_results['dbp_me']
        })
        
        print(f"Group {group_key} results:")
        print(f"  SBP MAE: {test_results['sbp_mae']:.2f} ± {test_results['sbp_std']:.2f}")
        print(f"  SBP RMSE: {test_results['sbp_rmse']:.2f}")
        print(f"  SBP ME: {test_results['sbp_me']:.2f}")
        print(f"  DBP MAE: {test_results['dbp_mae']:.2f} ± {test_results['dbp_std']:.2f}")
        print(f"  DBP RMSE: {test_results['dbp_rmse']:.2f}")
        print(f"  DBP ME: {test_results['dbp_me']:.2f}")
    
    # Calculate overall results and fairness
    results = {}
    if all_predictions:
        all_sbp_true = [p['sbp_true'] for p in all_predictions]
        all_sbp_pred = [p['sbp_pred'] for p in all_predictions]
        all_dbp_true = [p['dbp_true'] for p in all_predictions]
        all_dbp_pred = [p['dbp_pred'] for p in all_predictions]
        
        # Calculate errors
        sbp_errors = [abs(t - p) for t, p in zip(all_sbp_true, all_sbp_pred)]
        dbp_errors = [abs(t - p) for t, p in zip(all_dbp_true, all_dbp_pred)]
        
        # Signed errors for ME calculation
        sbp_signed_errors = [p - t for t, p in zip(all_sbp_true, all_sbp_pred)]
        dbp_signed_errors = [p - t for t, p in zip(all_dbp_true, all_dbp_pred)]
        
        overall_sbp_mae = np.mean(sbp_errors)
        overall_sbp_rmse = np.sqrt(np.mean([(t - p)**2 for t, p in zip(all_sbp_true, all_sbp_pred)]))
        overall_sbp_std = np.std(sbp_errors)
        overall_sbp_me = np.mean(sbp_signed_errors)
        
        overall_dbp_mae = np.mean(dbp_errors)
        overall_dbp_rmse = np.sqrt(np.mean([(t - p)**2 for t, p in zip(all_dbp_true, all_dbp_pred)]))
        overall_dbp_std = np.std(dbp_errors)
        overall_dbp_me = np.mean(dbp_signed_errors)
        
        # Get average number of parameters
        avg_params = int(np.mean([r['num_params'] for r in group_results]))
        
        # Calculate fairness metrics
        predictions_df = pd.DataFrame(all_predictions)
        fairness_metrics = calculate_fairness_metrics(predictions_df, stratify_type)
        
        results[stratify_type] = {
            'num_parameters': avg_params,
            'sbp_mae': overall_sbp_mae,
            'sbp_rmse': overall_sbp_rmse,
            'sbp_std': overall_sbp_std,
            'sbp_me': overall_sbp_me,
            'dbp_mae': overall_dbp_mae,
            'dbp_rmse': overall_dbp_rmse,
            'dbp_std': overall_dbp_std,
            'dbp_me': overall_dbp_me,
            'group_details': group_results,
            'fairness_metrics': fairness_metrics
        }
        
        print(f"\nOverall {stratify_type} results:")
        print(f"  Average model parameters: {avg_params}")
        print(f"  SBP MAE: {overall_sbp_mae:.2f} ± {overall_sbp_std:.2f}")
        print(f"  SBP RMSE: {overall_sbp_rmse:.2f}")
        print(f"  SBP ME: {overall_sbp_me:.2f}")
        print(f"  DBP MAE: {overall_dbp_mae:.2f} ± {overall_dbp_std:.2f}")
        print(f"  DBP RMSE: {overall_dbp_rmse:.2f}")
        print(f"  DBP ME: {overall_dbp_me:.2f}")
        print(f"  Number of groups: {len(group_results)}")
        
        # Print fairness metrics
        print(f"\nFairness metrics:")
        for metric_name, metric_value in fairness_metrics.items():
            print(f"  {metric_name}: {metric_value}")
        
        # Save detailed predictions to CSV
        predictions_file = os.path.join(
            output_dir, f'{run_name}_{stratify_type}_detailed_predictions.csv'
        )
        predictions_df.to_csv(predictions_file, index=False)
        print(f"Saved detailed predictions to: {predictions_file}")
    
    return results

def run_experiment(data_dir, fold_number, model_type, output_dir, batch_size=128, d_model=16, 
                  nhead=4, num_encoder_layers=2, dim_feedforward=64, input_signals='ppg_ecg',
                  num_epochs=100, data_format='subject_cv', train_measurements=None,
                  test_measurements=None, run_name=None):
    """Run experiment for specified model type"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Input signals: {input_signals}")
    
    # Load data
    data = prepare_data(
        data_dir,
        fold_number=fold_number,
        data_format=data_format,
        train_measurements=train_measurements,
        test_measurements=test_measurements
    )

    if run_name is None:
        if data_format == 'measurements':
            train_tag = '-'.join(str(value) for value in train_measurements)
            test_tag = '-'.join(str(value) for value in test_measurements)
            run_name = f'measurements_train_{train_tag}_test_{test_tag}'
        else:
            run_name = f'fold_{fold_number}'
    
    # Handle stratified models
    if model_type.startswith('stratified'):
        return run_stratified_model(
            data, model_type, run_name, output_dir,
            batch_size, d_model, nhead, num_encoder_layers, dim_feedforward,
            input_signals, num_epochs
        )
    
    # Define feature columns for each model type
    if model_type == 'base':
        feature_columns = []
    elif model_type == 'full_hypertension':
        feature_columns = ['history of hypertension']
    elif model_type == 'full_all':
        feature_columns = ['history of hypertension', 'age', 'gender', 'height', 'weight']
    
    results = {}
    
    # Regular training for non-stratified models
    print(f"\n=== Training {model_type} model ===")
    
    # Prepare feature scaler
    feature_scaler = prepare_feature_scaler(data['train_features'], feature_columns)
    
    # Create datasets
    train_dataset = SignalDataset(
        data['train_ecg'], data['train_ppg'], data['train_features'],
        data['train_sbp'], data['train_dbp'],
        feature_scaler=feature_scaler,
        feature_columns=feature_columns,
        input_signals=input_signals
    )
    
    test_dataset = SignalDataset(
        data['test_ecg'], data['test_ppg'], data['test_features'],
        data['test_sbp'], data['test_dbp'],
        signal_scaler=train_dataset.signal_scaler,
        feature_scaler=feature_scaler,
        feature_columns=feature_columns,
        target_stats=train_dataset.target_stats,
        input_signals=input_signals
    )
    
    # Split training data for validation
    val_size = int(0.2 * len(train_dataset))
    train_size = len(train_dataset) - val_size
    train_subset, val_subset = torch.utils.data.random_split(train_dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Create model
    num_features = len(feature_columns) if feature_columns else 0
    model = BPTransformer(
        d_model=d_model,
        nhead=nhead,
        num_encoder_layers=num_encoder_layers,
        dim_feedforward=dim_feedforward,
        num_features=num_features,
        input_signals=input_signals
    )
    
    # Count parameters
    num_params = count_parameters(model)
    print(f"Model parameters: {num_params}")
    
    # Train model
    model = train_model(
        model, train_loader, val_loader, train_dataset.target_stats,
        num_epochs=num_epochs, device=device
    )
    
    # Evaluate model
    test_results = evaluate_model_with_details(
        model, test_loader, data['test_features'], train_dataset.target_stats, device=device
    )
    
    # Create predictions dataframe with metadata
    predictions_df = pd.DataFrame(test_results['predictions'])
    predictions_df['model_type'] = model_type
    
    # Calculate fairness metrics
    fairness_metrics = calculate_fairness_metrics(predictions_df, model_type)
    
    results[model_type] = {
        'num_parameters': num_params,
        'sbp_mae': test_results['sbp_mae'],
        'sbp_rmse': test_results['sbp_rmse'],
        'sbp_std': test_results['sbp_std'],
        'sbp_me': test_results['sbp_me'],
        'dbp_mae': test_results['dbp_mae'],
        'dbp_rmse': test_results['dbp_rmse'],
        'dbp_std': test_results['dbp_std'],
        'dbp_me': test_results['dbp_me'],
        'fairness_metrics': fairness_metrics
    }
    
    print(f"Results for {model_type}:")
    print(f"  Model parameters: {num_params}")
    print(f"  SBP MAE: {test_results['sbp_mae']:.2f} ± {test_results['sbp_std']:.2f}")
    print(f"  SBP RMSE: {test_results['sbp_rmse']:.2f}")
    print(f"  SBP ME: {test_results['sbp_me']:.2f}")
    print(f"  DBP MAE: {test_results['dbp_mae']:.2f} ± {test_results['dbp_std']:.2f}")
    print(f"  DBP RMSE: {test_results['dbp_rmse']:.2f}")
    print(f"  DBP ME: {test_results['dbp_me']:.2f}")
    
    # Print fairness metrics
    print(f"\nFairness metrics:")
    for metric_name, metric_value in fairness_metrics.items():
        print(f"  {metric_name}: {metric_value}")
    
    # Save detailed predictions
    predictions_file = os.path.join(
        output_dir, f'{run_name}_{model_type}_detailed_predictions.csv'
    )
    predictions_df.to_csv(predictions_file, index=False)
    print(f"Saved detailed predictions to: {predictions_file}")
    
    return results
