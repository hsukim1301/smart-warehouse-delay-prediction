import pandas as pd
import numpy as np
import lightgbm as lgb
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import LabelEncoder
import os
import sys
import warnings
import gc

# Force UTF-8 encoding for stdout
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

warnings.filterwarnings('ignore')

# Set paths
DATA_DIR = r'C:\Users\김현수\Desktop\study\데이콘\지연 예측\warehouse_project\open'
TRAIN_PATH = os.path.join(DATA_DIR, 'train.csv')
TEST_PATH = os.path.join(DATA_DIR, 'test.csv')
LAYOUT_PATH = os.path.join(DATA_DIR, 'layout_info.csv')

def reduce_mem_usage(df):
    for col in df.columns:
        col_type = df[col].dtype
        if pd.api.types.is_numeric_dtype(col_type):
            c_min, c_max = df[col].min(), df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max: df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max: df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max: df[col] = df[col].astype(np.int32)
                else: df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max: df[col] = df[col].astype(np.float32)
                else: df[col] = df[col].astype(np.float64)
    return df

def preprocess_data(train, test, layout):
    print("Preprocessing data (Exp 45: Cumulative Max/Min vs Prev)...")
    train = train.merge(layout, on='layout_id', how='left')
    test = test.merge(layout, on='layout_id', how='left')
    
    le = LabelEncoder()
    all_layout_types = pd.concat([train['layout_type'], test['layout_type']])
    le.fit(all_layout_types.astype(str))
    train['layout_type'] = le.transform(train['layout_type'].astype(str))
    test['layout_type'] = le.transform(test['layout_type'].astype(str))
    
    for df in [train, test]:
        df['robot_utilization'] = df['robot_active'] / (df['robot_total'] + 1e-6)
        df['charger_utilization'] = df['robot_charging'] / (df['charger_count'] + 1e-6)
        df['aisle_pressure'] = df['congestion_score'] / (df['aisle_width_avg'] + 1e-6)
        df['intersection_density'] = df['intersection_count'] / (df['floor_area_sqm'] + 1e-6)
        df['pack_station_pressure'] = df['order_inflow_15m'] / (df['pack_station_count'] + 1e-6)
        df['bottleneck_risk'] = df['congestion_score'] * df['intersection_density'] / (df['aisle_width_avg'] + 1e-6)
        df['task_intensity'] = df['order_inflow_15m'] / (df['robot_active'] + 1e-6)

    layout_num_cols = ['aisle_width_avg', 'intersection_count', 'robot_total']
    key_ops = ['congestion_score', 'robot_active', 'order_inflow_15m']
    for l_col in layout_num_cols:
        for o_col in key_ops:
            train[f'{l_col}_x_{o_col}'] = train[l_col] * train[o_col]
            test[f'{l_col}_x_{o_col}'] = test[l_col] * test[o_col]

    momentum_cols = ['congestion_score', 'low_battery_ratio', 'robot_active']
    for col in momentum_cols:
        train[f'{col}_vel'] = train.groupby('scenario_id')[col].diff(1)
        test[f'{col}_vel'] = test.groupby('scenario_id')[col].diff(1)

    train['time_idx'] = train.groupby('scenario_id').cumcount()
    test['time_idx'] = test.groupby('scenario_id').cumcount()
    train = train.sort_values(['scenario_id', 'time_idx'])
    test = test.sort_values(['scenario_id', 'time_idx'])
    
    # [FEATURES] Full 22 SEQ_COLS Change vs Start (Exp 44)
    target_cols = [
        'order_inflow_15m', 'unique_sku_15m', 'robot_active', 'robot_idle',
        'robot_charging', 'battery_mean', 'battery_std', 'low_battery_ratio',
        'charge_queue_length', 'avg_charge_wait', 'congestion_score',
        'max_zone_density', 'blocked_path_15m', 'near_collision_15m',
        'fault_count_15m', 'avg_recovery_time', 'task_reassign_15m',
        'replenishment_overlap', 'pack_utilization', 'loading_dock_util',
        'staging_area_util', 'label_print_queue'
    ]
    for col in target_cols:
        first_val_tr = train.groupby('scenario_id')[col].transform('first')
        train[f'{col}_vs_start'] = train[col] / (first_val_tr + 1e-6)
        train[f'{col}_delta_start'] = train[col] - first_val_tr
        
        first_val_ts = test.groupby('scenario_id')[col].transform('first')
        test[f'{col}_vs_start'] = test[col] / (first_val_ts + 1e-6)
        test[f'{col}_delta_start'] = test[col] - first_val_ts

    # [NEW FEATURES] Cumulative Max/Min vs Current (Exp 45)
    for col in target_cols:
        if col not in train.columns:
            continue
        
        # shift(1)로 현재 값 누수 방지
        for df in [train, test]:
            prev = df.groupby('scenario_id')[col].shift(1)
            
            cum_max = prev.groupby(df['scenario_id']).cummax()
            cum_min = prev.groupby(df['scenario_id']).cummin()
            
            df[f'{col}_vs_cummax'] = df[col] / (cum_max + 1e-6)
            df[f'{col}_vs_cummin'] = df[col] / (cum_min.abs() + 1e-6)

    SEQ_COLS = ["order_inflow_15m", "unique_sku_15m", "robot_active", "low_battery_ratio", "charge_queue_length", "congestion_score", "fault_count_15m"]
    train_new_scen = train['scenario_id'].values != np.roll(train['scenario_id'].values, 1); train_new_scen[0] = True
    test_new_scen = test['scenario_id'].values != np.roll(test['scenario_id'].values, 1); test_new_scen[0] = True

    for col in SEQ_COLS:
        for lag in [1, 2]:
            tr_lag = train[col].shift(lag).values.copy(); ts_lag = test[col].shift(lag).values.copy()
            for l in range(lag):
                tr_lag[np.roll(train_new_scen, l)] = np.nan; ts_lag[np.roll(test_new_scen, l)] = np.nan
            train[f'{col}_lag{lag}'] = tr_lag; test[f'{col}_lag{lag}'] = ts_lag
        
        train[f'{col}_exp_mean'] = train.groupby('scenario_id')[col].transform(lambda x: x.shift(1).expanding().mean())
        test[f'{col}_exp_mean'] = test.groupby('scenario_id')[col].transform(lambda x: x.shift(1).expanding().mean())

    train['time_ratio'] = train.groupby('scenario_id')['time_idx'].transform(lambda x: x / (x.max() + 1e-6))
    test['time_ratio'] = test.groupby('scenario_id')['time_idx'].transform(lambda x: x / (x.max() + 1e-6))
    
    for df in [train, test]:
        df['congestion_ratio'] = df['congestion_score'] / (df['congestion_score_exp_mean'] + 1e-6)
        df['steps_remaining'] = df.groupby('scenario_id')['time_idx'].transform('max') - df['time_idx']

    train.fillna(0, inplace=True); test.fillna(0, inplace=True)
    return reduce_mem_usage(train), reduce_mem_usage(test)

def apply_smoothed_te(df_tr, df_val, target_col, k=30):
    global_mean = df_tr[target_col].mean()
    agg = df_tr.groupby('layout_id')[target_col].agg(['mean', 'std', 'median', 'count']).reset_index()
    agg['layout_mean'] = (agg['count'] * agg['mean'] + k * global_mean) / (agg['count'] + k)
    agg.rename(columns={'std': 'layout_std', 'median': 'layout_median', 'count': 'layout_count'}, inplace=True)
    df_val = df_val.merge(agg[['layout_id', 'layout_mean', 'layout_std', 'layout_median', 'layout_count']], on='layout_id', how='left')
    df_tr = df_tr.merge(agg[['layout_id', 'layout_mean', 'layout_std', 'layout_median', 'layout_count']], on='layout_id', how='left')
    df_val['layout_mean'] = df_val['layout_mean'].fillna(global_mean)
    df_val['layout_std'] = df_val['layout_std'].fillna(df_tr[target_col].std())
    df_val['layout_median'] = df_val['layout_median'].fillna(df_tr[target_col].median())
    df_val['layout_count'] = df_val['layout_count'].fillna(0)
    return df_tr, df_val, ['layout_mean', 'layout_std', 'layout_median', 'layout_count']

if __name__ == "__main__":
    print("--- Experiment 45: Cumulative Max/Min vs Prev ---")
    train_raw = pd.read_csv(TRAIN_PATH); test_raw = pd.read_csv(TEST_PATH); layout = pd.read_csv(LAYOUT_PATH)
    common_layouts = set(train_raw['layout_id'].unique()) & set(test_raw['layout_id'].unique())
    
    train, test = preprocess_data(train_raw, test_raw, layout)
    TARGET = 'avg_delay_minutes_next_30m'
    features_base = [c for c in train.columns if c not in ['ID', 'layout_id', 'scenario_id', TARGET]]
    train['is_seen'] = train['layout_id'].isin(common_layouts)

    # Parameters
    best_cat_params = {
        'iterations': 1441, 'learning_rate': 0.024382726628741795, 'depth': 7, 
        'l2_leaf_reg': 4.329713228202991, 'bagging_temperature': 0.15517607913494932,
        'random_seed': 42, 'loss_function': 'MAE', 'eval_metric': 'MAE', 'verbose': False, 'task_type': 'CPU'
    }
    lgb_params = {'learning_rate': 0.03, 'n_estimators': 1500, 'max_depth': 7, 'num_leaves': 63, 'random_state': 42, 'verbose': -1, 'objective': 'huber'}
    xgb_params = {'learning_rate': 0.03, 'n_estimators': 1000, 'max_depth': 7, 'random_state': 42, 'objective': 'reg:absoluteerror', 'verbosity': 0}

    gkf = GroupKFold(n_splits=5)
    oof_lgb = np.zeros(len(train)); oof_xgb = np.zeros(len(train)); oof_cat = np.zeros(len(train))

    for fold, (tr_idx, val_idx) in enumerate(gkf.split(train, train[TARGET], groups=train['layout_id'])):
        print(f"\n--- Fold {fold+1} ---")
        X_tr, y_tr = train.iloc[tr_idx].copy(), train.iloc[tr_idx][TARGET]
        X_val, y_val = train.iloc[val_idx].copy(), train.iloc[val_idx][TARGET]
        
        X_tr, X_val, te_cols = apply_smoothed_te(X_tr, X_val, TARGET, k=30)
        X_tr.fillna(0, inplace=True); X_val.fillna(0, inplace=True)
        feats = features_base + te_cols
        
        # LGBM
        m_lgb = LGBMRegressor(**lgb_params).fit(X_tr[feats], np.log1p(y_tr), eval_set=[(X_val[feats], np.log1p(y_val))], callbacks=[lgb.early_stopping(100)])
        oof_lgb[val_idx] = np.expm1(m_lgb.predict(X_val[feats]))
        
        # XGB
        m_xgb = XGBRegressor(**xgb_params).fit(X_tr[feats], np.log1p(y_tr))
        oof_xgb[val_idx] = np.expm1(m_xgb.predict(X_val[feats]))
        
        # CatBoost
        m_cat = CatBoostRegressor(**best_cat_params).fit(X_tr[feats], np.log1p(y_tr), eval_set=[(X_val[feats], np.log1p(y_val))], early_stopping_rounds=100)
        oof_cat[val_idx] = np.expm1(m_cat.predict(X_val[feats]))
        
        print(f"Fold {fold+1} MAE - LGB: {mean_absolute_error(y_val, oof_lgb[val_idx]):.4f}, XGB: {mean_absolute_error(y_val, oof_xgb[val_idx]):.4f}, CAT: {mean_absolute_error(y_val, oof_cat[val_idx]):.4f}")

    # Inverse MAE Weighting
    mae_lgb = mean_absolute_error(train[TARGET], oof_lgb)
    mae_xgb = mean_absolute_error(train[TARGET], oof_xgb)
    mae_cat = mean_absolute_error(train[TARGET], oof_cat)
    
    inv_mae_sum = (1/mae_lgb) + (1/mae_xgb) + (1/mae_cat)
    w_lgb = (1/mae_lgb) / inv_mae_sum
    w_xgb = (1/mae_xgb) / inv_mae_sum
    w_cat = (1/mae_cat) / inv_mae_sum
    
    print(f"\n--- Ensemble Weights (Inverse MAE) ---")
    print(f"LGB (MAE {mae_lgb:.4f}): {w_lgb:.4f}")
    print(f"XGB (MAE {mae_xgb:.4f}): {w_xgb:.4f}")
    print(f"CAT (MAE {mae_cat:.4f}): {w_cat:.4f}")

    oof_ens = (oof_lgb * w_lgb) + (oof_xgb * w_xgb) + (oof_cat * w_cat)
    total_mae = mean_absolute_error(train[TARGET], oof_ens)
    seen_mae = mean_absolute_error(train[train['is_seen']][TARGET], oof_ens[train['is_seen']])
    unseen_mae = mean_absolute_error(train[~train['is_seen']][TARGET], oof_ens[~train['is_seen']])

    print(f"\nTotal OOF MAE: {total_mae:.4f} (Seen: {seen_mae:.4f}, Unseen: {unseen_mae:.4f})")

    # Final Test Prediction
    print("\nTraining Final Models for Submission...")
    full_train_agg, full_test_agg, te_cols = apply_smoothed_te(train, test, TARGET, k=30)
    full_train_agg.fillna(0, inplace=True); full_test_agg.fillna(0, inplace=True)
    feats = features_base + te_cols
    
    final_lgb = LGBMRegressor(**lgb_params).fit(full_train_agg[feats], np.log1p(train[TARGET]))
    final_xgb = XGBRegressor(**xgb_params).fit(full_train_agg[feats], np.log1p(train[TARGET]))
    final_cat = CatBoostRegressor(**best_cat_params).fit(full_train_agg[feats], np.log1p(train[TARGET]))
    
    p_lgb = np.expm1(final_lgb.predict(full_test_agg[feats]))
    p_xgb = np.expm1(final_xgb.predict(full_test_agg[feats]))
    p_cat = np.expm1(final_cat.predict(full_test_agg[feats]))
    
    test_preds = (p_lgb * w_lgb) + (p_xgb * w_xgb) + (p_cat * w_cat)
    test_preds = np.maximum(test_preds, 0)
                  
    pd.DataFrame({'ID': test['ID'], TARGET: test_preds}).to_csv('submission_45.csv', index=False)
    
    print("\nExperiment 45 complete. Submission saved as submission_45.csv")
