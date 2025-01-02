

# 生成数据
#%% Data generation Y->X
import os
import sys
# 设置环境变量，必须在导入 tensorflow 之前
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
pid = os.getpid()
print(f"Current PID: {pid}")
import argparse
import numpy as np
import nonlincausality as nlc
import optuna
import pandas as pd
import logging
import tensorflow as tf
# from optuna.integration import TFKerasPruningCallback
import concurrent.futures
# tf.config.experimental.enable_tensor_float_32_execution(False)

# 配置 logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger()
logger.setLevel(logging.INFO)
# 只显示ERROR级别的日志
# optuna.logging.set_verbosity(optuna.logging.ERROR)

# 数据集划分比例---global
train_percent = 0.6
val_percent = 0
test_percent = 0.4
df_col = {
    'emission': 0,
    'ever': 1,
    'phev': 2,
    'fuel': 3,
    'ev': 4
}
parser = argparse.ArgumentParser()
parser.add_argument('--province', type=str, default='anhui',
                    help='province name for data loading')
args = parser.parse_args()
def prepare_data(emission_data, vehicle_data):

    # 数据日期清洗和准备
    emission_data['date'] = emission_data['date'].astype(str)
    vehicle_data['date'] = vehicle_data['date'].astype(str)
    emission_data['date'] = pd.to_datetime(emission_data['date'], format='%B %Y')  
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
    return merged_data_n


# 定义相关函数
def data_selection(merged_data_n, fuel_col_idx):
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

    return data_train, data_val, data_test


def compute_error(data_test, model_X, model_XY, lag):
    from nonlincausality.utils import prepare_data_for_prediction, calculate_pred_and_errors

    data_X, data_XY = prepare_data_for_prediction(data_test, lag)
    # X_pred_X = model_X.predict(data_X, verbose=0)
    # X_pred_XY = model_XY.predict(data_XY, verbose=0)
    # error_X = np.mean((data_test[lag:, 0] - X_pred_X) ** 2)
    # error_XY = np.mean((data_test[lag:, 0] - X_pred_XY) ** 2)
    current_x = data_test[lag:, 0]
    _, _, error_X, error_XY = calculate_pred_and_errors(
    current_x,
    data_X,
    data_XY,
    model_X,
    model_XY,
    )
    # print("debug_here")
    # print(data_test[:, 0].shape)
    # print(data_test[lag:, 0].shape)
    # print(X_pred_X.shape)
    return error_X, error_XY

# 相关主目标函数
def objective(trial, merged_data_n, fuel_col_idx, lags=[2]):
    # # 获取当前 trial 的编号
    # trial_number = trial.number
    # # 获取当前 trial 的开始时间
    # start_time = trial.datetime_start
    data_train, data_val, data_test = data_selection(merged_data_n, fuel_col_idx)
    # # 输出当前 trial 的信息
    # print(f"启动第 {trial_number} 个 trial，开始时间为 {start_time}")
    # print(data_train.shape, data_val.shape, data_test.shape)


    n_layers = 2
    NN_config = ['g', 'dr']
    NN_neurons = []

    # # 根据输入数据大小调整神经元数量
    # input_size = data_train.shape[1]  # 获取输入维度
    # max_neurons = min(512, input_size * 4)  # 限制最大神经元数量

    for i in range(n_layers):
        if i % 2 == 0:
            NN_neurons.append(trial.suggest_int(f'n_units_{i}', 4, 64, step=4))
        else:
            NN_neurons.append(trial.suggest_float(f'dropout_{i}', 0.1, 0.3, step=0.05))

    # callbacks = [
    #     TFKerasPruningCallback(trial, 'val_loss'),
    #     tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    # ]
    # 添加数据shape检查
    # logger.info(f"Data shapes - Train: {data_train.shape}, Val: {data_val.shape}, Test: {data_test.shape}")
    # logger.info(f"Network architecture: {NN_neurons}")
    params = {
        'x': data_train,
        'maxlag': lags,
        'NN_config': NN_config,
        'NN_neurons': NN_neurons,
        'x_test': data_test,
        'run': trial.suggest_int('run', 1, 3, step=1),
        'epochs_num': [trial.suggest_int('epochs_1', 30, 100, step=10), trial.suggest_int('epochs_2', 20, 100, step=10)],
        'learning_rate': [trial.suggest_float('lr_1', 1e-4, 1e-2, log=True), trial.suggest_float('lr_2', 1e-5, 1e-3, log=True)],
        'batch_size_num': trial.suggest_int('batch_size', 8, 16, step=4),
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

    params['callbacks'] = None
    params['verbose'] = False
    params['plot'] = False

    try:
        results = nlc.nonlincausalityNN(**params)
        total_err = 0
        for lag in lags:
            model_X = results[lag].best_model_X
            model_XY = results[lag].best_model_XY
            X_err, XY_err = compute_error(data_test, model_X, model_XY, lag)
            # 只进行 X 预测的 MSE 计算 
            total_err += np.mean(X_err)  # 累加误差
        avg_err = total_err / len(lags)
        return avg_err
    except Exception as e:
        logger.error(f"Trial {trial.number} failed: {e}")
        return float('inf')
        #########################################

    finally:
        tf.keras.backend.clear_session()  # 清理会话


def run_study(study, objective, data, df_col, n_trials=20, n_jobs=1, gpus=None):
    
    if gpus:
        logging.info(f"GPUs available: {gpus}")
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)
    else:
        logging.info("No GPUs available")
        tf.config.threading.set_intra_op_parallelism_threads(n_jobs)
        tf.config.threading.set_inter_op_parallelism_threads(n_jobs)
        logging.info(f"CPU 配置完成，使用 {n_jobs} 个线程")
            
    try:
        study.optimize(lambda trial: objective(trial, data, df_col), n_trials, n_jobs)
        return {
            'best_params': study.best_params if study.best_trial else None,
            'best_value': study.best_value if study.best_trial else None
        }
    except Exception as e:
        print(f"Study optimization failed: {e}")
        return None

# # 在创建 study 之前调用
def create_study(name):
    return optuna.create_study(
        study_name=f"{name}_study",
        direction='minimize',
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10, interval_steps=1),
        sampler=optuna.samplers.TPESampler(seed=42)
    )

if __name__ == "__main__":

    try:
        gpus = tf.config.list_physical_devices('GPU')
        # args.province = parser.parse_known_args()[0]
        # 获取省份名称首字母大写形式
        province_cap = args.province.capitalize()
        # 读取数据
        emission_data = pd.read_csv('dataset/total_monthly_state_emission.csv')
        selected_columns = emission_data[['num', 'date']].copy()  # 使用列名选择
        selected_columns[province_cap] = emission_data[province_cap]  # 添加省份列
        # 更新emission_data
        emission_data = selected_columns
        vehicle_data = pd.read_csv(f'dataset/car/result_{args.province}.csv')  
        merged_data_n = prepare_data(emission_data, vehicle_data)
        logging.debug(merged_data_n)
        if merged_data_n is None:
            raise ValueError("Data preparation failed")

        # 创建学习对象时添加 study_name
        study_ever = optuna.create_study(
            study_name="EVER_STUDY",  # 添加名称
            direction='minimize',
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10, interval_steps=1),
            sampler=optuna.samplers.TPESampler(seed=42)
        )
        study_ev = optuna.create_study(
            study_name="EV_STUDY",    # 添加名称
            direction='minimize',
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10, interval_steps=1),
            sampler=optuna.samplers.TPESampler(seed=42)
        )
        study_phev = optuna.create_study(
            study_name="PHEV_STUDY",  # 添加名称
            direction='minimize',
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10, interval_steps=1),
            sampler=optuna.samplers.TPESampler(seed=42)
        )
        study_fuel = optuna.create_study(
            study_name="FUEL_STUDY",  # 添加名称
            direction='minimize',
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10, interval_steps=1),
            sampler=optuna.samplers.TPESampler(seed=42)
        )

        results = {}
        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures = {
                'ever': executor.submit(run_study, study_ever, objective, merged_data_n, df_col['ever']),
                'ev': executor.submit(run_study, study_ev, objective, merged_data_n, df_col['ev']), 
                'phev': executor.submit(run_study, study_phev, objective, merged_data_n, df_col['phev']),
                'fuel': executor.submit(run_study, study_fuel, objective, merged_data_n, df_col['fuel'])
            }  
            for name, future in futures.items():
                try:
                    results[name] = future.result()
                except Exception as e:
                    print(f"Error in {name} study: {e}")
                    results[name] = None
        # 打印结果
        for name, result in results.items():
            if result:
                print(f"\n{name} Study Best parameters:")
                print(result['best_params'])
                print(f"Best value (Average MSE):", result['best_value'])
            else:
                print(f"\n{name} Study failed to complete")
    except Exception as e:
        logger.error(f"Program execution failed: {str(e)}")
        sys.exit(1)


    
#%%