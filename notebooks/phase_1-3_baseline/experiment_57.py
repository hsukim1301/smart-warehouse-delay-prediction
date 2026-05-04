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
    print("Preprocessing data (Exp 57: Exp 52 Base + Exp 56 Pruning)...")
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

    for col in target_cols:
        if col not in train.columns:
            continue
        for df in [train, test]:
            prev = df.groupby('scenario_id')[col].shift(1)
            cum_max = prev.groupby(df['scenario_id']).cummax()
            cum_min = prev.groupby(df['scenario_id']).cummin()
            df[f'{col}_vs_cummax'] = df[col] / (cum_max + 1e-6)
            df[f'{col}_vs_cummin'] = df[col] / (cum_min.abs() + 1e-6)

    for col in target_cols:
        if col not in train.columns:
            continue
        for df in [train, test]:
            prev = df.groupby('scenario_id')[col].shift(1)
            cum_max = prev.groupby(df['scenario_id']).cummax()
            cum_min = prev.groupby(df['scenario_id']).cummin()
            cum_range = cum_max - cum_min
            df[f'{col}_position_in_range'] = ((df[col] - cum_min) / (cum_range + 1e-3)).clip(0, 2)

    # [EXP 48 FEATURES MAINTAINED]
    target_lag_cols = [
        'congestion_score', 'fault_count_15m', 'charge_queue_length',
        'low_battery_ratio', 'blocked_path_15m', 'avg_recovery_time',
        'near_collision_15m', 'pack_utilization'
    ]
    for col in target_lag_cols:
        if col not in train.columns:
            continue
        for df in [train, test]:
            for lag in [4, 5, 6, 7]:
                df[f'{col}_lag{lag}'] = df.groupby('scenario_id')[col].shift(lag)
            prev = df.groupby('scenario_id')[col].shift(1)
            grp = prev.groupby(df['scenario_id'])
            df[f'{col}_roll7_mean'] = grp.rolling(7, min_periods=1).mean().reset_index(level=0, drop=True)
            df[f'{col}_roll7_std'] = grp.rolling(7, min_periods=1).std().reset_index(level=0, drop=True)
            df[f'{col}_roll10_mean'] = grp.rolling(10, min_periods=1).mean().reset_index(level=0, drop=True)
            df[f'{col}_roll10_std'] = grp.rolling(10, min_periods=1).std().reset_index(level=0, drop=True)

    SEQ_COLS_BASE = ["order_inflow_15m", "unique_sku_15m", "robot_active", "low_battery_ratio", "charge_queue_length", "congestion_score", "fault_count_15m"]
    
    # [EXP 49 FEATURES MAINTAINED]
    remaining_cols = [c for c in SEQ_COLS_BASE if c not in target_lag_cols]
    for col in remaining_cols:
        for df in [train, test]:
            for lag in [4, 5]:
                df[f'{col}_lag{lag}'] = df.groupby('scenario_id')[col].shift(lag)
            prev = df.groupby('scenario_id')[col].shift(1)
            grp = prev.groupby(df['scenario_id'])
            df[f'{col}_roll7_mean'] = grp.rolling(7, min_periods=1).mean().reset_index(level=0, drop=True)
            df[f'{col}_roll7_std']  = grp.rolling(7, min_periods=1).std().reset_index(level=0, drop=True)

    # [EXP 50 FEATURES MAINTAINED]
    for col in target_cols:
        for df in [train, test]:
            for lag in [8, 10]:
                df[f'{col}_lag{lag}'] = df.groupby('scenario_id')[col].shift(lag)

    for col in target_lag_cols:
        for df in [train, test]:
            prev = df.groupby('scenario_id')[col].shift(1)
            grp = prev.groupby(df['scenario_id'])
            df[f'{col}_roll14_mean'] = grp.rolling(14, min_periods=1).mean().reset_index(level=0, drop=True)
            df[f'{col}_roll14_std']  = grp.rolling(14, min_periods=1).std().reset_index(level=0, drop=True)

    # [EXP 51 FEATURES MAINTAINED]
    SEQ_COLS = target_cols 
    for col in SEQ_COLS:
        for df in [train, test]:
            for lag in [12, 15]:
                df[f'{col}_lag{lag}'] = df.groupby('scenario_id')[col].shift(lag)

    for col in target_lag_cols:
        for df in [train, test]:
            prev = df.groupby('scenario_id')[col].shift(1)
            grp = prev.groupby(df['scenario_id'])
            df[f'{col}_roll20_mean'] = grp.rolling(20, min_periods=1).mean().reset_index(level=0, drop=True)
            df[f'{col}_roll20_std']  = grp.rolling(20, min_periods=1).std().reset_index(level=0, drop=True)

    # [EXP 52 FEATURES MAINTAINED]
    for col in SEQ_COLS:
        for df in [train, test]:
            for lag in [20, 24]:
                df[f'{col}_lag{lag}'] = df.groupby('scenario_id')[col].shift(lag)

    remaining_cols_exp52 = [c for c in SEQ_COLS if c not in target_lag_cols]
    for col in remaining_cols_exp52:
        for df in [train, test]:
            prev = df.groupby('scenario_id')[col].shift(1)
            grp = prev.groupby(df['scenario_id'])
            df[f'{col}_roll14_mean'] = grp.rolling(14, min_periods=1).mean().reset_index(level=0, drop=True)
            df[f'{col}_roll14_std']  = grp.rolling(14, min_periods=1).std().reset_index(level=0, drop=True)
            df[f'{col}_roll20_mean'] = grp.rolling(20, min_periods=1).mean().reset_index(level=0, drop=True)
            df[f'{col}_roll20_std']  = grp.rolling(20, min_periods=1).std().reset_index(level=0, drop=True)

    train_new_scen = train['scenario_id'].values != np.roll(train['scenario_id'].values, 1); train_new_scen[0] = True
    test_new_scen = test['scenario_id'].values != np.roll(test['scenario_id'].values, 1); test_new_scen[0] = True

    for col in SEQ_COLS_BASE:
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
    print("--- Experiment 57: Exp 52 + Pruning + 10-Seed Ensemble ---")
    train_raw = pd.read_csv(TRAIN_PATH); test_raw = pd.read_csv(TEST_PATH); layout = pd.read_csv(LAYOUT_PATH)
    common_layouts = set(train_raw['layout_id'].unique()) & set(test_raw['layout_id'].unique())
    
    train, test = preprocess_data(train_raw, test_raw, layout)
    TARGET = 'avg_delay_minutes_next_30m'
    features_base = [c for c in train.columns if c not in ['ID', 'layout_id', 'scenario_id', TARGET, 'is_seen']]
    train['is_seen'] = train['layout_id'].isin(common_layouts)

    # 107 Zero-Importance Features from Exp 56
    zero_imp_features = [
        'charge_queue_length', 'avg_charge_wait', 'charge_queue_length_lag2', 'fault_count_15m_lag2',
        'time_ratio', 'task_reassign_15m_vs_cummin', 'low_battery_ratio_vel', 'low_battery_ratio_lag2',
        'task_reassign_15m', 'blocked_path_15m_lag8', 'blocked_path_15m_lag10', 'near_collision_15m_lag8',
        'near_collision_15m_lag10', 'fault_count_15m_lag8', 'fault_count_15m_lag10', 'avg_recovery_time_lag8',
        'avg_recovery_time_lag10', 'task_reassign_15m_lag8', 'task_reassign_15m_lag10', 'replenishment_overlap_lag8',
        'replenishment_overlap_lag10', 'robot_charging_lag10', 'low_battery_ratio_lag8', 'low_battery_ratio_lag10',
        'charge_queue_length_lag8', 'charge_queue_length_lag10', 'avg_charge_wait_lag8', 'avg_charge_wait_lag10',
        'fault_count_15m_vs_cummax', 'fault_count_15m_vs_cummin', 'avg_recovery_time_vs_cummax', 'task_reassign_15m_vs_cummax',
        'low_battery_ratio_vs_cummax', 'low_battery_ratio_vs_cummin', 'charge_queue_length_vs_cummin', 'avg_charge_wait_vs_cummax',
        'blocked_path_15m_vs_cummax', 'near_collision_15m_vs_cummin', 'charge_queue_length_roll14_mean', 'task_reassign_15m_vs_start',
        'avg_recovery_time_lag5', 'avg_recovery_time_lag6', 'avg_recovery_time_lag7', 'near_collision_15m_lag4',
        'near_collision_15m_lag5', 'near_collision_15m_lag6', 'near_collision_15m_lag7', 'charge_queue_length_roll7_std',
        'charge_queue_length_roll10_mean', 'low_battery_ratio_lag4', 'low_battery_ratio_lag5', 'low_battery_ratio_lag7',
        'blocked_path_15m_lag4', 'blocked_path_15m_lag5', 'blocked_path_15m_lag6', 'blocked_path_15m_lag7',
        'label_print_queue_delta_start', 'robot_charging_lag15', 'low_battery_ratio_lag12', 'low_battery_ratio_lag15',
        'charge_queue_length_lag12', 'charge_queue_length_lag15', 'avg_charge_wait_lag12', 'avg_charge_wait_lag15',
        'congestion_score_lag12', 'max_zone_density_lag15', 'blocked_path_15m_lag12', 'fault_count_15m_lag4',
        'fault_count_15m_lag5', 'fault_count_15m_lag6', 'fault_count_15m_lag7', 'charge_queue_length_lag4',
        'charge_queue_length_lag5', 'charge_queue_length_lag6', 'charge_queue_length_lag7', 'charge_queue_length_roll7_mean',
        'blocked_path_15m_lag15', 'near_collision_15m_lag12', 'near_collision_15m_lag15', 'fault_count_15m_lag12',
        'fault_count_15m_lag15', 'avg_recovery_time_lag12', 'avg_recovery_time_lag15', 'task_reassign_15m_lag12',
        'task_reassign_15m_lag15', 'replenishment_overlap_lag12', 'replenishment_overlap_lag15', 'charge_queue_length_position_in_range',
        'avg_charge_wait_position_in_range', 'congestion_score_position_in_range', 'blocked_path_15m_position_in_range', 'near_collision_15m_position_in_range',
        'fault_count_15m_position_in_range', 'avg_recovery_time_position_in_range', 'task_reassign_15m_position_in_range', 'replenishment_overlap_position_in_range',
        'label_print_queue_position_in_range', 'replenishment_overlap_vs_cummin', 'staging_area_util_vs_cummax', 'battery_mean_position_in_range',
        'low_battery_ratio_position_in_range', 'label_print_queue_lag15', 'charge_queue_length_roll20_std', 'charge_queue_length_vs_start',
        'charge_queue_length_delta_start', 'avg_charge_wait_vs_start', 'avg_charge_wait_delta_start'
    ]
    
    features_pruned = [f for f in features_base if f not in zero_imp_features]
    print(f"Features reduced from {len(features_base)} to {len(features_pruned)} (-{len(features_base)-len(features_pruned)})")

    seeds = [42, 123, 2026, 777, 1004, 314, 555, 888, 999, 1337]
    
    cat_params_base = {
        'iterations': 1441, 'learning_rate': 0.024382726628741795, 'depth': 7, 
        'l2_leaf_reg': 4.329713228202991, 'bagging_temperature': 0.15517607913494932,
        'loss_function': 'MAE', 'eval_metric': 'MAE', 'verbose': False, 'task_type': 'CPU',
        'early_stopping_rounds': 100
    }
    lgb_params_base = {'learning_rate': 0.03, 'n_estimators': 1500, 'max_depth': 7, 'num_leaves': 63, 'verbose': -1, 'objective': 'huber'}
    xgb_params_base = {'learning_rate': 0.03, 'n_estimators': 1000, 'max_depth': 7, 'objective': 'reg:absoluteerror', 'verbosity': 0, 'early_stopping_rounds': 100}

    gkf = GroupKFold(n_splits=5)
    
    oof_seed_ensembles = np.zeros(len(train))
    test_preds_total = np.zeros(len(test))

    for seed_idx, seed in enumerate(seeds):
        print(f"\n{'='*30} Seed {seed} ({seed_idx+1}/10) {'='*30}")
        
        oof_lgb = np.zeros(len(train))
        oof_xgb = np.zeros(len(train))
        oof_cat = np.zeros(len(train))
        
        # Store models for final prediction if needed, but we'll do final training on full data per seed logic or just simple average
        # To save memory and time, we'll accumulate test predictions during folds or train on full data once per seed.
        # Given "10-Seed Ensemble Strategy", we'll do 5-fold CV for EACH seed to get OOF and weights.
        
        seed_lgb_preds = np.zeros(len(test))
        seed_xgb_preds = np.zeros(len(test))
        seed_cat_preds = np.zeros(len(test))

        for fold, (tr_idx, val_idx) in enumerate(gkf.split(train, train[TARGET], groups=train['layout_id'])):
            X_tr, y_tr = train.iloc[tr_idx].copy(), train.iloc[tr_idx][TARGET]
            X_val, y_val = train.iloc[val_idx].copy(), train.iloc[val_idx][TARGET]
            
            X_tr, X_val, te_cols = apply_smoothed_te(X_tr, X_val, TARGET, k=30)
            X_tr.fillna(0, inplace=True); X_val.fillna(0, inplace=True)
            feats = features_pruned + te_cols
            
            # Use fold-specific test set with TE
            _, X_test_fold, _ = apply_smoothed_te(X_tr, test.copy(), TARGET, k=30)
            X_test_fold.fillna(0, inplace=True)
            
            # LGBM
            m_lgb = LGBMRegressor(**{**lgb_params_base, 'random_state': seed}).fit(
                X_tr[feats], np.log1p(y_tr), 
                eval_set=[(X_val[feats], np.log1p(y_val))], 
                callbacks=[lgb.early_stopping(100)]
            )
            oof_lgb[val_idx] = np.expm1(m_lgb.predict(X_val[feats]))
            seed_lgb_preds += np.expm1(m_lgb.predict(X_test_fold[feats])) / 5
            
            # XGB
            m_xgb = XGBRegressor(**{**xgb_params_base, 'random_state': seed}).fit(
                X_tr[feats], np.log1p(y_tr), 
                eval_set=[(X_val[feats], np.log1p(y_val))], 
                verbose=False
            )
            oof_xgb[val_idx] = np.expm1(m_xgb.predict(X_val[feats]))
            seed_xgb_preds += np.expm1(m_xgb.predict(X_test_fold[feats])) / 5
            
            # CatBoost
            m_cat = CatBoostRegressor(**{**cat_params_base, 'random_seed': seed}).fit(
                X_tr[feats], np.log1p(y_tr), 
                eval_set=[(X_val[feats], np.log1p(y_val))]
            )
            oof_cat[val_idx] = np.expm1(m_cat.predict(X_val[feats]))
            seed_cat_preds += np.expm1(m_cat.predict(X_test_fold[feats])) / 5
            
            print(f"Fold {fold+1} Done", end=' | ', flush=True)
        
        # Calculate weights for this seed
        mae_lgb = mean_absolute_error(train[TARGET], oof_lgb)
        mae_xgb = mean_absolute_error(train[TARGET], oof_xgb)
        mae_cat = mean_absolute_error(train[TARGET], oof_cat)
        
        inv_mae_sum = (1/mae_lgb) + (1/mae_xgb) + (1/mae_cat)
        w_lgb, w_xgb, w_cat = (1/mae_lgb)/inv_mae_sum, (1/mae_xgb)/inv_mae_sum, (1/mae_cat)/inv_mae_sum
        
        seed_oof = (oof_lgb * w_lgb) + (oof_xgb * w_xgb) + (oof_cat * w_cat)
        oof_seed_ensembles += seed_oof / 10
        
        seed_test_preds = (seed_lgb_preds * w_lgb) + (seed_xgb_preds * w_xgb) + (seed_cat_preds * w_cat)
        test_preds_total += seed_test_preds / 10
        
        print(f"\nSeed {seed} MAE: {mean_absolute_error(train[TARGET], seed_oof):.4f} (LGB:{mae_lgb:.4f}, XGB:{mae_xgb:.4f}, CAT:{mae_cat:.4f})", flush=True)
        gc.collect()

    # Final Performance Analysis
    total_mae = mean_absolute_error(train[TARGET], oof_seed_ensembles)
    seen_mae = mean_absolute_error(train[train['is_seen']][TARGET], oof_seed_ensembles[train['is_seen']])
    unseen_mae = mean_absolute_error(train[~train['is_seen']][TARGET], oof_seed_ensembles[~train['is_seen']])

    print("\n" + "=" * 65)
    print(f"  [Exp 57 Heavy Ensemble Result]")
    print(f"  - Seeds used      : {len(seeds)}")
    print(f"  - Features Pruned : {len(zero_imp_features)}")
    print(f"  - Total OOF MAE   : {total_mae:.4f}")
    print(f"  - Seen MAE        : {seen_mae:.4f}")
    print(f"  - Unseen MAE      : {unseen_mae:.4f} (Exp 52: 8.6732)")
    print(f"  - Improvement     : {8.7493 - total_mae:+.4f} (vs Exp 52 Baseline)")
    print("=" * 65)

    test_preds_total = np.maximum(test_preds_total, 0)
    pd.DataFrame({'ID': test['ID'], TARGET: test_preds_total}).to_csv('submission_57.csv', index=False)
    print("\nExperiment 57 complete. Submission saved as submission_57.csv")
