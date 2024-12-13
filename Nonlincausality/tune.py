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

# 生成数据
#%% Data generation Y->X
# 读取数据
emission_data = pd.read_csv('dataset/total_monthly_state_emission.csv')
vehicle_data = pd.read_csv('dataset/result_anhui.csv')
# 数据日期清洗和准备
# 首先，你需要确保日期列是字符串类型
emission_data['date'] = emission_data['date'].astype(str)
vehicle_data['date'] = vehicle_data['date'].astype(str)
# 使用pd.to_datetime将日期字符串转换为日期时间对象
# format参数指定了原始日期字符串的格式Jan-19
emission_data['date'] = pd.to_datetime(emission_data['date'], format='%b-%y')
vehicle_data['date'] = pd.to_datetime(vehicle_data['date'], format='%Y/%m/%d')
# 将日期时间对象转换回字符串，格式为'YYYY-MM-DD'
emission_data['date'] = emission_data['date'].dt.strftime('%Y-%m-%d')
vehicle_data['date'] = vehicle_data['date'].dt.strftime('%Y-%m-%d')
# 选取时间范围
start_date = '2022-06-01'
end_date = '2024-09-01'
vehicle_data_mask = (vehicle_data['date'] >= start_date) & (vehicle_data['date'] <= end_date)
emission_data_mask = (emission_data['date'] >= start_date) & (emission_data['date'] <= end_date)

emission_data = emission_data.loc[emission_data_mask]
vehicle_data = vehicle_data.loc[vehicle_data_mask]
# 丢掉第一列
emission_data=emission_data.drop(emission_data.columns[[0]],axis=1)
# 数据对齐
emission_data.set_index('date', inplace=True)
vehicle_data.set_index('date', inplace=True)

# logging.debug(emission_data)
# logging.debug(vehicle_data)
# Merge Data by Date
# Group vehicle_data by date and sum the values for all cities
vehicle_data_grouped = vehicle_data.groupby('date').sum()

# Merge the emission_data and vehicle_data_grouped datasets on the date column
merged_data = pd.merge(emission_data, vehicle_data_grouped, on='date', how='inner')

# Display the merged data
merged_data=merged_data.drop(merged_data.columns[[1]],axis=1)
merged_data

#%%
# 准备数据集
from sklearn.preprocessing import StandardScaler
# 创建归一化器
scaler = StandardScaler()

# 归一化数据
'''
FIXME: 一起做归一化之后出现大量极值
'''
merged_data_n = scaler.fit_transform(merged_data)
data_train = merged_data_n[:16, [1, 0]]
data_val = merged_data_n[16:22, [1, 0]]
data_test = merged_data_n[22:, [1, 0]]
#%%
import tensorflow as tf

# 定义训练函数
lags = [3]
def objective(trial):
    # 定义网络层数
    n_layers = 2 # trial.suggest_int('n_layers', 2, 4)
    # lags = [3]#trial.suggest_int('lags', 2, 4)
    
    # 构建网络配置
    NN_config = ['g', 'dr']
    NN_neurons = []
    
    for i in range(n_layers):
        # 交替使用Dense和Dropout层
        if i % 2 == 0:
            # NN_config.append('d')
            NN_neurons.append(trial.suggest_int(f'n_units_{i}', 32, 256))
        else:
            # NN_config.append('dr')
            NN_neurons.append(trial.suggest_float(f'dropout_{i}', 0.01, 0.3))
    callbacks = [
        TFKerasPruningCallback(trial, 'val_loss'),  # 使用验证损失作为剪枝指标
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
    ]
    # 超参数配置
    params = {
        'x': data_train,
        'maxlag': lags,
        'NN_config': NN_config,
        'NN_neurons': NN_neurons,
        'x_test': data_test,
        'run': trial.suggest_int('run', 2, 5),
        'epochs_num': [
            trial.suggest_int('epochs_1', 30, 200),
            trial.suggest_int('epochs_2', 20, 100)
        ],
        'learning_rate': [
            trial.suggest_float('lr_1', 1e-4, 1e-2, log=True),
            trial.suggest_float('lr_2', 1e-5, 1e-3, log=True)
        ],
        'batch_size_num': trial.suggest_int('batch_size', 4, 64),
        'x_val': data_val,
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-5, 1e-2, log=True),
        'callbacks': callbacks,
        'verbose': False,
        'plot': False
    }
    
    try:
        # 运行模型
        results = nlc.nonlincausalityNN(**params)
        
        # 计算所有lag的平均误差作为优化目标
        total_error = 0
        for lag in lags:
            best_errors_X = results[lag].best_errors_X
            best_errors_XY = results[lag].best_errors_XY
            mse = np.mean(best_errors_X**2)  # 使用X预测的MSE作为评估指标
            total_error += mse
            
        avg_error = total_error / len(lags)
        
        return avg_error
        
    except Exception as e:
        print(f"Trial failed: {e}")
        return float('inf')

# 创建 study 对象时使用 MedianPruner
study = optuna.create_study(
    direction='minimize',
    pruner=optuna.pruners.MedianPruner(
        n_startup_trials=5,        # 在开始剪枝之前完成的试验数
        n_warmup_steps=10,         # 在开始剪枝之前的训练步数
        interval_steps=1           # 进行剪枝检查的间隔步数
    ),
    sampler=optuna.samplers.TPESampler(seed=42)
)

# 开始优化
study.optimize(objective, n_trials=50, n_jobs=1)  # 可以根据需要调整trials数量

# 打印最佳参数
print("\nBest parameters:")
print(study.best_params)
print("\nBest value (Average MSE):", study.best_value)
#%%
# # 使用最佳参数运行最终模型
# best_params = study.best_params

# # 构建最佳网络配置
# best_n_layers = best_params['n_layers']
# best_NN_config = []
# best_NN_neurons = []

# for i in range(best_n_layers):
#     if i % 2 == 0:
#         best_NN_config.append('d')
#         best_NN_neurons.append(best_params[f'n_units_{i}'])
#     else:
#         best_NN_config.append('dr')
#         best_NN_neurons.append(best_params[f'dropout_{i}'])

# # 运行最终模型
# final_results = nlc.nonlincausalityNN(
#     x=data_train,
#     maxlag=lags,
#     NN_config=best_NN_config,
#     NN_neurons=best_NN_neurons,
#     x_test=data_test,
#     run=best_params['run'],
#     epochs_num=[best_params['epochs_1'], best_params['epochs_2']],
#     learning_rate=[best_params['lr_1'], best_params['lr_2']],
#     batch_size_num=best_params['batch_size'],
#     x_val=data_val,
#     reg_alpha=best_params['reg_alpha'],
#     callbacks=None,
#     verbose=True,
#     plot=True
# )

# # 打印最终结果
# for lag in lags:
#     p_value = final_results[lag].p_value
#     test_statistic = final_results[lag]._test_statistic
#     print(f"\nFor lag {lag}:")
#     print(f"Test statistic = {test_statistic}")
#     print(f"p-value = {p_value}")
#%%