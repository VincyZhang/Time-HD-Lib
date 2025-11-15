
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
class Dataset_Custom(Dataset):
    def __init__(self, args, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', seasonal_patterns=None):
        # size [seq_len, label_len, pred_len]
        self.args = args
        # info
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
#
        #dict_raw = load_dataset("Time-HD-Anonymous/High_Dimensional_Time_Series", self.args.data, cache_dir="dataset")
        #
        #df_raw = dict_raw['train'].to_pandas()
        #'''
        #df_raw.columns: ['date', ...(other features), target feature]
        #'''
        #cols = list(df_raw.columns)
        #cols.remove('date')
        #df_raw = df_raw[['date'] + cols]
        #num_train = int(len(df_raw) * 0.7)
        #num_test = int(len(df_raw) * 0.2)
        #num_vali = len(df_raw) - num_train - num_test
        #border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        #border2s = [num_train, num_train + num_vali, len(df_raw)]
        #border1 = border1s[self.set_type]
        #border2 = border2s[self.set_type]
#
        #if self.features == 'M' or self.features == 'MS':
        #    cols_data = df_raw.columns[1:]
        #    df_data = df_raw[cols_data]
        #elif self.features == 'S':
        #    df_data = df_raw[[self.target]]
#
        #if self.scale:
        #    train_data = df_data[border1s[0]:border2s[0]]
        #    self.scaler.fit(train_data.values)
        #    data = self.scaler.transform(df_data.values)
        #else:
        #    data = df_data.values
#
        #df_stamp = df_raw[['date']][border1:border2]
        #df_stamp['date'] = pd.to_datetime(df_stamp.date)
        #if self.timeenc == 0:
        #    df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
        #    df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
        #    df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
        #    df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
        #    data_stamp = df_stamp.drop(['date'], 1).values
        #elif self.timeenc == 1:
        #    data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
        #    data_stamp = data_stamp.transpose(1, 0)
#
        #self.data_x = data[border1:border2]
        #self.data_y = data[border1:border2]
#
        ## Safely handle data augmentation
        #augmentation_ratio = getattr(self.args, 'augmentation_ratio', 0)
        #if self.set_type == 0 and augmentation_ratio > 0:
        #    self.data_x, self.data_y, augmentation_tags = run_augmentation_single(self.data_x, self.data_y, self.args)
#
        #self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

from core.config import BaseConfig
config_dict = {
    "model_name": "U-Cast",
    "dataset_name": "custom",
    "expand": 2,
    "d_conv": 4,
    "channel_reduction_ratio": 16,
    # Model architecture
    "d_model": 512,
    "n_heads": 8,
    "e_layers": 2,
    "d_layers": 1,
    "d_ff": 2048,
    "dropout": 0.1,
    "activation": "gelu",
    "lradj": "type3",
    # Dataset related settings
    # Input/Output dimensions
    "enc_in": 1161,
    "dec_in": 7,
    "c_out": 7,
    # Training parameters
    "learning_rate": 0.0005,
    "batch_size": 32,
    "train_epochs": 100,
    "patience": 3,
    "alpha": 0.0,
    # Task specific
    #seq_len = pred_len * seq_len_factor
    "seq_len_factor": 3,
    "pred_len": 96,
    "seq_len": 288,
    "label_len": 0,
    "task_name": "long_term_forecast",
    # Time features
    "freq": "h",  # frequency: h for hourly, d for daily, etc.
    # Moving average for decomposition
    "moving_avg": 25
}

config = BaseConfig.from_dict(config_dict)

# Use the core data provider implementation
data_set = Dataset_Custom(
    config = config,
    size=[config.seq_len, config.label_len, config.pred_len])
data_loader = DataLoader(
    data_set,
    batch_size=config.batch_size,
    shuffle=True,
    num_workers=1,
    drop_last=False,
    pin_memory=True)


from core.experiments.long_term_forecasting import LongTermForecastingExperiment
# fix seed
import random
seed = 2025
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# init runner
from accelerate import Accelerator
runner = LongTermForecastingExperiment(config)
# enable mixed precision to reduce Mem Usage
runner.accelerator = Accelerator(mixed_precision="bf16")
runner.device = runner.accelerator.device
# Execute training phase
setting = "Ucast_custom_test"
results = {}
all_epoch_metrics, best_metrics, best_model_path = runner.train(setting)

results['training'] = {
    'all_epoch_metrics': all_epoch_metrics,
    'best_metrics': best_metrics,
    'best_model_path': best_model_path
}

import gc
torch.cuda.empty_cache()
gc.collect()
torch.cuda.ipc_collect()
if torch.distributed.is_initialized():
    torch.distributed.barrier()
    
# Test with the best trained model
runner.accelerator.print(f'>>> Starting testing for {setting} <<<')
mse, mae = runner.test(setting, best_model_path)

results['testing'] = {
    'mse': mse,
    'mae': mae
}