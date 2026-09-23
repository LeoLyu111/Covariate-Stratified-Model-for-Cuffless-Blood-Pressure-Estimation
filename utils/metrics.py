import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

def calculate_regression_fairness(sbp_true, sbp_pred, dbp_true, dbp_pred, groups):
    """
    Calculate regression fairness metric:
    fairness = |mean(y_true_group1) - mean(y_true_group2)| / |mean(y_pred_group1) - mean(y_pred_group2)|
    """
    unique_groups = np.unique(groups)
    if len(unique_groups) != 2:
        return None, None
    
    group1_mask = groups == unique_groups[0]
    group2_mask = groups == unique_groups[1]
    
    # SBP fairness
    sbp_true_diff = abs(np.mean(np.array(sbp_true)[group1_mask]) - np.mean(np.array(sbp_true)[group2_mask]))
    sbp_pred_diff = abs(np.mean(np.array(sbp_pred)[group1_mask]) - np.mean(np.array(sbp_pred)[group2_mask]))
    sbp_fairness = sbp_true_diff / sbp_pred_diff if sbp_pred_diff != 0 else float('inf')
    
    # DBP fairness
    dbp_true_diff = abs(np.mean(np.array(dbp_true)[group1_mask]) - np.mean(np.array(dbp_true)[group2_mask]))
    dbp_pred_diff = abs(np.mean(np.array(dbp_pred)[group1_mask]) - np.mean(np.array(dbp_pred)[group2_mask]))
    dbp_fairness = dbp_true_diff / dbp_pred_diff if dbp_pred_diff != 0 else float('inf')
    
    return sbp_fairness, dbp_fairness

def classify_bp(sbp, dbp):
    """
    Classify BP into categories:
    0: Normal (SBP < 120 and DBP < 80)
    1: Elevated/Hypertension (SBP >= 120 or DBP >= 80)
    """
    return ((np.array(sbp) >= 120) | (np.array(dbp) >= 80)).astype(int)

def calculate_demographic_parity(sbp_true, sbp_pred, dbp_true, dbp_pred, groups):
    """
    Calculate demographic parity based on BP classification
    """
    unique_groups = np.unique(groups)
    if len(unique_groups) != 2:
        return None, None
    
    # Classify BP values
    true_class = classify_bp(sbp_true, dbp_true)
    pred_class = classify_bp(sbp_pred, dbp_pred)
    
    group1_mask = groups == unique_groups[0]
    group2_mask = groups == unique_groups[1]
    
    # Calculate positive rates for each group (predicted as hypertensive)
    group1_pos_rate = np.mean(pred_class[group1_mask])
    group2_pos_rate = np.mean(pred_class[group2_mask])
    
    # Demographic parity violation (absolute difference in positive rates)
    demographic_parity_diff = abs(group1_pos_rate - group2_pos_rate)
    
    return demographic_parity_diff, {
        'group1_pos_rate': group1_pos_rate,
        'group2_pos_rate': group2_pos_rate,
        'group1_name': unique_groups[0],
        'group2_name': unique_groups[1]
    }

def calculate_fairness_metrics(predictions_df, model_type):
    """Calculate fairness metrics based on model type"""
    fairness_results = {}
    
    if model_type in ['base', 'full_hypertension', 'full_all']:
        # For non-stratified models, calculate fairness by gender and hypertension
        for group_attr in ['gender', 'history_of_hypertension']:
            if group_attr in predictions_df.columns:
                groups = predictions_df[group_attr].values
                
                # Regression fairness
                sbp_reg_fair, dbp_reg_fair = calculate_regression_fairness(
                    predictions_df['sbp_true'].values,
                    predictions_df['sbp_pred'].values,
                    predictions_df['dbp_true'].values,
                    predictions_df['dbp_pred'].values,
                    groups
                )
                
                # Demographic parity
                demo_parity, demo_details = calculate_demographic_parity(
                    predictions_df['sbp_true'].values,
                    predictions_df['sbp_pred'].values,
                    predictions_df['dbp_true'].values,
                    predictions_df['dbp_pred'].values,
                    groups
                )
                
                fairness_results[f'{group_attr}_regression'] = {
                    'sbp_fairness': sbp_reg_fair,
                    'dbp_fairness': dbp_reg_fair
                }
                fairness_results[f'{group_attr}_demographic_parity'] = {
                    'parity_difference': demo_parity,
                    'details': demo_details
                }
    
    elif model_type == 'stratified_hypertension':
        # Calculate fairness between hypertension groups
        if 'hypertension_group' in predictions_df.columns:
            groups = predictions_df['hypertension_group'].values
            
            # Regression fairness
            sbp_reg_fair, dbp_reg_fair = calculate_regression_fairness(
                predictions_df['sbp_true'].values,
                predictions_df['sbp_pred'].values,
                predictions_df['dbp_true'].values,
                predictions_df['dbp_pred'].values,
                groups
            )
            
            # Demographic parity
            demo_parity, demo_details = calculate_demographic_parity(
                predictions_df['sbp_true'].values,
                predictions_df['sbp_pred'].values,
                predictions_df['dbp_true'].values,
                predictions_df['dbp_pred'].values,
                groups
            )
            
            fairness_results['hypertension_regression'] = {
                'sbp_fairness': sbp_reg_fair,
                'dbp_fairness': dbp_reg_fair
            }
            fairness_results['hypertension_demographic_parity'] = {
                'parity_difference': demo_parity,
                'details': demo_details
            }
    
    elif model_type == 'stratified_gender':
        # Calculate fairness between gender groups
        if 'gender_group' in predictions_df.columns:
            groups = predictions_df['gender_group'].values
            
            # Regression fairness
            sbp_reg_fair, dbp_reg_fair = calculate_regression_fairness(
                predictions_df['sbp_true'].values,
                predictions_df['sbp_pred'].values,
                predictions_df['dbp_true'].values,
                predictions_df['dbp_pred'].values,
                groups
            )
            
            # Demographic parity
            demo_parity, demo_details = calculate_demographic_parity(
                predictions_df['sbp_true'].values,
                predictions_df['sbp_pred'].values,
                predictions_df['dbp_true'].values,
                predictions_df['dbp_pred'].values,
                groups
            )
            
            fairness_results['gender_regression'] = {
                'sbp_fairness': sbp_reg_fair,
                'dbp_fairness': dbp_reg_fair
            }
            fairness_results['gender_demographic_parity'] = {
                'parity_difference': demo_parity,
                'details': demo_details
            }
    
    elif model_type == 'stratified_hypertension_gender':
        # Calculate fairness for both hypertension and gender, then take mean
        hypertension_results = {}
        gender_results = {}
        
        # Fairness by hypertension
        if 'hypertension_group' in predictions_df.columns:
            hyper_groups = predictions_df['hypertension_group'].values
            sbp_reg_fair_h, dbp_reg_fair_h = calculate_regression_fairness(
                predictions_df['sbp_true'].values,
                predictions_df['sbp_pred'].values,
                predictions_df['dbp_true'].values,
                predictions_df['dbp_pred'].values,
                hyper_groups
            )
            demo_parity_h, demo_details_h = calculate_demographic_parity(
                predictions_df['sbp_true'].values,
                predictions_df['sbp_pred'].values,
                predictions_df['dbp_true'].values,
                predictions_df['dbp_pred'].values,
                hyper_groups
            )
        
        # Fairness by gender
        if 'gender_group' in predictions_df.columns:
            gender_groups = predictions_df['gender_group'].values
            sbp_reg_fair_g, dbp_reg_fair_g = calculate_regression_fairness(
                predictions_df['sbp_true'].values,
                predictions_df['sbp_pred'].values,
                predictions_df['dbp_true'].values,
                predictions_df['dbp_pred'].values,
                gender_groups
            )
            demo_parity_g, demo_details_g = calculate_demographic_parity(
                predictions_df['sbp_true'].values,
                predictions_df['sbp_pred'].values,
                predictions_df['dbp_true'].values,
                predictions_df['dbp_pred'].values,
                gender_groups
            )
        
        # Calculate mean fairness
        if 'hypertension_group' in predictions_df.columns and 'gender_group' in predictions_df.columns:
            mean_sbp_reg_fair = np.mean([sbp_reg_fair_h, sbp_reg_fair_g])
            mean_dbp_reg_fair = np.mean([dbp_reg_fair_h, dbp_reg_fair_g])
            mean_demo_parity = np.mean([demo_parity_h, demo_parity_g])
            
            fairness_results['hypertension_regression'] = {
                'sbp_fairness': sbp_reg_fair_h,
                'dbp_fairness': dbp_reg_fair_h
            }
            fairness_results['gender_regression'] = {
                'sbp_fairness': sbp_reg_fair_g,
                'dbp_fairness': dbp_reg_fair_g
            }
            fairness_results['hypertension_demographic_parity'] = {
                'parity_difference': demo_parity_h,
                'details': demo_details_h
            }
            fairness_results['gender_demographic_parity'] = {
                'parity_difference': demo_parity_g,
                'details': demo_details_g
            }
            fairness_results['mean_regression'] = {
                'sbp_fairness': mean_sbp_reg_fair,
                'dbp_fairness': mean_dbp_reg_fair
            }
            fairness_results['mean_demographic_parity'] = {
                'parity_difference': mean_demo_parity
            }
    
    return fairness_results