import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error, mean_squared_error


def _calculate_regression_monitor(targets, predictions):
    """Calculate diagnostics that reveal regression-to-the-mean collapse."""
    targets = np.asarray(targets, dtype=np.float64).reshape(-1)
    predictions = np.asarray(predictions, dtype=np.float64).reshape(-1)

    target_mean = float(np.mean(targets))
    prediction_mean = float(np.mean(predictions))
    target_std = float(np.std(targets))
    prediction_std = float(np.std(predictions))

    if len(targets) > 1 and target_std > 0 and prediction_std > 0:
        correlation = float(np.corrcoef(targets, predictions)[0, 1])
    else:
        correlation = float('nan')

    target_sum_squares = float(np.sum((targets - target_mean) ** 2))
    if target_sum_squares > 0:
        r2 = 1.0 - float(np.sum((targets - predictions) ** 2)) / target_sum_squares
    else:
        r2 = float('nan')

    std_ratio = prediction_std / target_std if target_std > 0 else float('nan')

    return {
        'target_mean': target_mean,
        'target_std': target_std,
        'prediction_mean': prediction_mean,
        'prediction_std': prediction_std,
        'prediction_min': float(np.min(predictions)),
        'prediction_max': float(np.max(predictions)),
        'correlation': correlation,
        'r2': r2,
        'std_ratio': std_ratio
    }


def _print_regression_monitor(name, metrics):
    print(
        f"  {name}: "
        f"target={metrics['target_mean']:.2f}+/-{metrics['target_std']:.2f}, "
        f"pred={metrics['prediction_mean']:.2f}+/-{metrics['prediction_std']:.2f}, "
        f"pred_range=[{metrics['prediction_min']:.2f}, {metrics['prediction_max']:.2f}], "
        f"std_ratio={metrics['std_ratio']:.3f}, "
        f"corr={metrics['correlation']:.3f}, R2={metrics['r2']:.3f}"
    )

    if np.isfinite(metrics['std_ratio']) and metrics['std_ratio'] < 0.1:
        print(
            f"  WARNING: {name} prediction variation is below 10% of target variation; "
            "possible mean-prediction collapse."
        )


def _inverse_transform(values, target_name, target_stats):
    values = np.asarray(values, dtype=np.float64)
    return (
        values * target_stats[f'{target_name}_std']
        + target_stats[f'{target_name}_mean']
    )


def train_model(model, train_loader, val_loader, target_stats, num_epochs=100,
                learning_rate=0.001, device='cuda'):
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    
    best_val_loss = float('inf')
    patience_counter = 0
    early_stopping_patience = 20
    shape_check_reported = False

    print(
        "Target standardization (training-set statistics): "
        f"SBP mean={target_stats['sbp_mean']:.2f}, std={target_stats['sbp_std']:.2f}; "
        f"DBP mean={target_stats['dbp_mean']:.2f}, std={target_stats['dbp_std']:.2f}"
    )
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        
        for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
            signals = batch['signals'].to(device)
            sbp_target = batch['sbp'].to(device)
            dbp_target = batch['dbp'].to(device)
            features = batch.get('features', None)
            if features is not None:
                features = features.to(device)
            
            optimizer.zero_grad()
            
            sbp_pred, dbp_pred = model(signals, features)

            assert sbp_pred.shape == sbp_target.shape, (
                f"SBP shape mismatch: prediction={tuple(sbp_pred.shape)}, "
                f"target={tuple(sbp_target.shape)}. This may cause incorrect broadcasting."
            )
            assert dbp_pred.shape == dbp_target.shape, (
                f"DBP shape mismatch: prediction={tuple(dbp_pred.shape)}, "
                f"target={tuple(dbp_target.shape)}. This may cause incorrect broadcasting."
            )

            if not shape_check_reported:
                print(
                    f"Shape check passed: SBP pred/target={tuple(sbp_pred.shape)}, "
                    f"DBP pred/target={tuple(dbp_pred.shape)}"
                )
                shape_check_reported = True
            
            sbp_loss = criterion(sbp_pred, sbp_target)
            dbp_loss = criterion(dbp_pred, dbp_target)
            loss = sbp_loss + dbp_loss
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_sbp_targets = []
        val_sbp_predictions = []
        val_dbp_targets = []
        val_dbp_predictions = []
        
        with torch.no_grad():
            for batch in val_loader:
                signals = batch['signals'].to(device)
                sbp_target = batch['sbp'].to(device)
                dbp_target = batch['dbp'].to(device)
                features = batch.get('features', None)
                if features is not None:
                    features = features.to(device)
                
                sbp_pred, dbp_pred = model(signals, features)

                assert sbp_pred.shape == sbp_target.shape, (
                    f"Validation SBP shape mismatch: prediction={tuple(sbp_pred.shape)}, "
                    f"target={tuple(sbp_target.shape)}."
                )
                assert dbp_pred.shape == dbp_target.shape, (
                    f"Validation DBP shape mismatch: prediction={tuple(dbp_pred.shape)}, "
                    f"target={tuple(dbp_target.shape)}."
                )
                
                sbp_loss = criterion(sbp_pred, sbp_target)
                dbp_loss = criterion(dbp_pred, dbp_target)
                loss = sbp_loss + dbp_loss
                
                val_loss += loss.item()

                val_sbp_targets.extend(sbp_target.detach().cpu().reshape(-1).numpy())
                val_sbp_predictions.extend(sbp_pred.detach().cpu().reshape(-1).numpy())
                val_dbp_targets.extend(dbp_target.detach().cpu().reshape(-1).numpy())
                val_dbp_predictions.extend(dbp_pred.detach().cpu().reshape(-1).numpy())
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

        val_sbp_targets_mmhg = _inverse_transform(val_sbp_targets, 'sbp', target_stats)
        val_sbp_predictions_mmhg = _inverse_transform(val_sbp_predictions, 'sbp', target_stats)
        val_dbp_targets_mmhg = _inverse_transform(val_dbp_targets, 'dbp', target_stats)
        val_dbp_predictions_mmhg = _inverse_transform(val_dbp_predictions, 'dbp', target_stats)

        sbp_monitor = _calculate_regression_monitor(
            val_sbp_targets_mmhg, val_sbp_predictions_mmhg
        )
        dbp_monitor = _calculate_regression_monitor(
            val_dbp_targets_mmhg, val_dbp_predictions_mmhg
        )
        print('Validation collapse monitor:')
        _print_regression_monitor('SBP', sbp_monitor)
        _print_regression_monitor('DBP', dbp_monitor)
        
        scheduler.step(val_loss)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= early_stopping_patience:
            print(f'Early stopping at epoch {epoch+1}')
            break
    
    return model

def evaluate_model_with_details(model, test_loader, test_features_df, target_stats, device='cuda'):
    """Enhanced evaluation function that returns detailed predictions with metadata"""
    model.eval()
    
    all_predictions = []
    
    with torch.no_grad():
        batch_idx = 0
        for batch in test_loader:
            signals = batch['signals'].to(device)
            sbp_target = batch['sbp'].to(device)
            dbp_target = batch['dbp'].to(device)
            features = batch.get('features', None)
            if features is not None:
                features = features.to(device)
            
            sbp_output, dbp_output = model(signals, features)

            assert sbp_output.shape == sbp_target.shape, (
                f"Test SBP shape mismatch: prediction={tuple(sbp_output.shape)}, "
                f"target={tuple(sbp_target.shape)}."
            )
            assert dbp_output.shape == dbp_target.shape, (
                f"Test DBP shape mismatch: prediction={tuple(dbp_output.shape)}, "
                f"target={tuple(dbp_target.shape)}."
            )

            sbp_target_mmhg = (
                sbp_target * target_stats['sbp_std'] + target_stats['sbp_mean']
            )
            sbp_output_mmhg = (
                sbp_output * target_stats['sbp_std'] + target_stats['sbp_mean']
            )
            dbp_target_mmhg = (
                dbp_target * target_stats['dbp_std'] + target_stats['dbp_mean']
            )
            dbp_output_mmhg = (
                dbp_output * target_stats['dbp_std'] + target_stats['dbp_mean']
            )
            
            # Get batch predictions
            batch_size = signals.size(0)
            for i in range(batch_size):
                # Calculate the actual index in the test set
                actual_idx = batch_idx * test_loader.batch_size + i
                if actual_idx < len(test_features_df):
                    # Get metadata from test features
                    row = test_features_df.iloc[actual_idx]
                    
                    sbp_true_val = sbp_target_mmhg[i].cpu().item()
                    sbp_pred_val = sbp_output_mmhg[i].cpu().item()
                    dbp_true_val = dbp_target_mmhg[i].cpu().item()
                    dbp_pred_val = dbp_output_mmhg[i].cpu().item()
                    
                    prediction_record = {
                        'test_index': actual_idx,
                        'subject_id': row['subject_id'],
                        'age': row['age'],
                        'gender': row['gender'],
                        'height': row['hight'],
                        'weight': row['weight'],
                        'history_of_hypertension': row['history of hypertension'],
                        'sbppre': row['sbppre'],
                        'sbppost': row['sbppost'],
                        'dbppre': row['dbppre'],
                        'dbppost': row['dbppost'],
                        'sbp_true': sbp_true_val,
                        'sbp_pred': sbp_pred_val,
                        'dbp_true': dbp_true_val,
                        'dbp_pred': dbp_pred_val,
                        'sbp_error': abs(sbp_true_val - sbp_pred_val),
                        'dbp_error': abs(dbp_true_val - dbp_pred_val),
                        'sbp_signed_error': sbp_pred_val - sbp_true_val,  # For ME calculation
                        'dbp_signed_error': dbp_pred_val - dbp_true_val   # For ME calculation
                    }
                    all_predictions.append(prediction_record)
            
            batch_idx += 1
    
    # Calculate overall metrics with standard deviation and mean error
    sbp_true = [p['sbp_true'] for p in all_predictions]
    sbp_pred = [p['sbp_pred'] for p in all_predictions]
    dbp_true = [p['dbp_true'] for p in all_predictions]
    dbp_pred = [p['dbp_pred'] for p in all_predictions]
    
    sbp_errors = [abs(t - p) for t, p in zip(sbp_true, sbp_pred)]
    dbp_errors = [abs(t - p) for t, p in zip(dbp_true, dbp_pred)]
    
    # Signed errors for ME calculation
    sbp_signed_errors = [p - t for t, p in zip(sbp_true, sbp_pred)]
    dbp_signed_errors = [p - t for t, p in zip(dbp_true, dbp_pred)]
    
    # Calculate metrics
    sbp_mae = mean_absolute_error(sbp_true, sbp_pred)
    sbp_rmse = np.sqrt(mean_squared_error(sbp_true, sbp_pred))
    sbp_std = np.std(sbp_errors)
    sbp_me = np.mean(sbp_signed_errors)  # Mean Error (ME)
    
    dbp_mae = mean_absolute_error(dbp_true, dbp_pred)
    dbp_rmse = np.sqrt(mean_squared_error(dbp_true, dbp_pred))
    dbp_std = np.std(dbp_errors)
    dbp_me = np.mean(dbp_signed_errors)  # Mean Error (ME)
    
    return {
        'predictions': all_predictions,
        'sbp_mae': sbp_mae,
        'sbp_rmse': sbp_rmse,
        'sbp_std': sbp_std,
        'sbp_me': sbp_me,
        'dbp_mae': dbp_mae,
        'dbp_rmse': dbp_rmse,
        'dbp_std': dbp_std,
        'dbp_me': dbp_me
    }
