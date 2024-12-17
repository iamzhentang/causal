

# 生成数据
#%% Data generation Y->X
import os
import numpy as np
import nonlincausality as nlc
import optuna
import pandas as pd
import logging
import tensorflow as tf
from optuna.integration import TFKerasPruningCallback

# 配置 logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger()
logger.setLevel(logging.INFO)
# 读取数据
emission_data = pd.read_csv('dataset/000/anhui/total_monthly_state_emission.csv')
vehicle_data = pd.read_csv('dataset/000/anhui/result_anhui.csv')
df_col = {
    'emission': 0,
    'EVER': 1,
    'PHEV': 2,
    'FUEL': 3,
    'EV': 4
}
fuel_col_idx = df_col['EVER']
# 数据集划分比例
train_percent = 0.6
val_percent = 0.2
test_percent = 0.2

# 数据日期清洗和准备
emission_data['date'] = emission_data['date'].astype(str)
vehicle_data['date'] = vehicle_data['date'].astype(str)
emission_data['date'] = pd.to_datetime(emission_data['date'], format='%b-%y')
# vehicle_data['date'] = pd.to_datetime(vehicle_data['date'], format='%Y/%m/%d')
emission_data['date'] = emission_data['date'].dt.strftime('%Y-%m-%d')
# vehicle_data['date'] = vehicle_data['date'].dt.strftime('%Y-%m-%d')

# 选取时间范围
start_date = '2022-06-01'
end_date = '2024-09-01'
vehicle_data_mask = (vehicle_data['date'] >= start_date) & (vehicle_data['date'] <= end_date)
emission_data_mask = (emission_data['date'] >= start_date) & (emission_data['date'] <= end_date)

emission_data = emission_data.loc[emission_data_mask]
vehicle_data = vehicle_data.loc[vehicle_data_mask]

# 丢掉第一列
emission_data = emission_data.drop(emission_data.columns[[0]], axis=1)

# 数据对齐
emission_data.set_index('date', inplace=True)
vehicle_data.set_index('date', inplace=True)
vehicle_data_grouped = vehicle_data.groupby('date').sum()
merged_data = pd.merge(emission_data, vehicle_data_grouped, on='date', how='inner')
merged_data = merged_data.drop(merged_data.columns[[1]], axis=1)
merged_data

# 数据模块化
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
merged_data_n = scaler.fit_transform(merged_data)
# 确保比例之和为1
assert train_percent + val_percent + test_percent == 1, "划分比例之和必须为1"

# 计算索引
total = len(merged_data_n)
train_end = int(total * train_percent)
val_end = int(train_end) + round(total * val_percent)

# 数据集划分
data_train = merged_data_n[:train_end, [fuel_col_idx, 0]]
data_val = merged_data_n[train_end:val_end, [fuel_col_idx, 0]]
data_test = merged_data_n[val_end:, [fuel_col_idx, 0]]
print(data_train.shape, data_val.shape, data_test.shape)
# data_train = merged_data_n[:16, [fuel_col_idx, 0]]
# data_val = merged_data_n[16:22, [fuel_col_idx, 0]]
# data_test = merged_data_n[22:, [fuel_col_idx, 0]]

# 定义相关函数
lags = [3]

def compute_error(data_test, model_X, model_XY, lag):
    from nonlincausality.utils import prepare_data_for_prediction, calculate_pred_and_errors
    data_X, data_XY = prepare_data_for_prediction(data_test, lag)
    X_pred_X = model_X.predict(data_X)
    X_pred_XY = model_XY.predict(data_XY)
    mse_X = np.mean((data_test[lag:, 0] - X_pred_X) ** 2)
    mse_XY = np.mean((data_test[lag:, 0] - X_pred_XY) ** 2)
    return mse_X, mse_XY

# 相关主目标函数
def objective(trial):
    n_layers = 2
    NN_config = ['g', 'dr']
    NN_neurons = []
    for i in range(n_layers):
        if i % 2 == 0:
            NN_neurons.append(trial.suggest_int(f'n_units_{i}', 64, 512, step=16))
        else:
            NN_neurons.append(trial.suggest_float(f'dropout_{i}', 0.1, 0.3, step=0.05))

    callbacks = [
        TFKerasPruningCallback(trial, 'val_loss'),
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    ]

    params = {
        'x': data_train,
        'maxlag': lags,
        'NN_config': NN_config,
        'NN_neurons': NN_neurons,
        'x_test': data_test,
        'run': trial.suggest_int('run', 1, 3, step=1),
        'epochs_num': [trial.suggest_int('epochs_1', 30, 100, step=10), trial.suggest_int('epochs_2', 20, 100, step=10)],
        'learning_rate': [trial.suggest_float('lr_1', 1e-4, 1e-2, log=True), trial.suggest_float('lr_2', 1e-5, 1e-3, log=True)],
        'batch_size_num': trial.suggest_int('batch_size', 1, 3, step=1),
        'x_val': data_val,
        'regularization': trial.suggest_categorical('regularization', ['l1_l2', 'l1', 'l2', 'None']),
    }
    # set numbers of reg_alpha_1 or reg_alpha_1 via l1 l2 config
    if params['regularization'] == 'l1_l2':
        params['reg_alpha'] = [
            trial.suggest_float('reg_alpha_1', 1e-5, 1e-2, log=True),
            trial.suggest_float('reg_alpha_2', 1e-5, 1e-2, log=True)
        ]
    elif params['regularization'] in ['l1', 'l2']:
        params['reg_alpha'] = trial.suggest_float('reg_alpha_1', 1e-5, 1e-2, log=True)
    else:
        params['reg_alpha'] = None

    params['callbacks'] = callbacks
    params['verbose'] = False
    params['plot'] = False

    try:
        results = nlc.nonlincausalityNN(**params)
        total_error = 0
        for lag in lags:
            model_X = results[lag].best_model_X
            model_XY = results[lag].best_model_XY
            mse_X, mse_XY = compute_error(data_test, model_X, model_XY, lag)
            total_error += mse_X  # 只进行 X 预测的 MSE 计算
        avg_error = total_error / len(lags)
        return avg_error
    except Exception as e:
        print(f"Trial failed: {e}")
        return float('inf')

# 创建学习对象
study = optuna.create_study(
    direction='minimize',
    pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10, interval_steps=1),
    sampler=optuna.samplers.TPESampler(seed=42)
)

# 开始优化
study.optimize(objective, n_trials=50, n_jobs=-1)

# 打印结果
print("\nBest parameters:")
print(study.best_params)
print("\nBest value (Average MSE):", study.best_value)
#%%