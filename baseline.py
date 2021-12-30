import os
import matplotlib.pyplot as plt
from datasets.elec_dataset import ElectricityDataModule
from models.normal import NormalGRU
import pytorch_lightning as pl
import lightgbm as lgb
from pytorch_lightning.callbacks import TQDMProgressBar, EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.metrics import mean_absolute_percentage_error
from utils.utils import get_array_data, inverse_norm
from config import config


def run_normal_GRU():
    dm = ElectricityDataModule()
    normal_GRU = NormalGRU()

    progress_bar, early_stop = TQDMProgressBar(refresh_rate=0), EarlyStopping(monitor='val_mape')
    checkpoint = ModelCheckpoint(monitor='val_mape',
                                 dirpath=config.mdoel_saving_dir,
                                 filename='NormalGRU-{epoch}-{step}-{val_mape:.5f}')
    tb_logger = TensorBoardLogger("tb_logs", name="NormalGRU")

    trainer = pl.Trainer(max_epochs=config.max_epochs,
                         min_epochs=config.min_epochs,
                         callbacks=[progress_bar, early_stop, checkpoint],
                         logger=[tb_logger],
                         gpus=1)
    trainer.fit(normal_GRU, datamodule=dm)
    test_results = trainer.test(normal_GRU, datamodule=dm, verbose=False)
    return test_results[0]['test_mape']


def run_lightgbm():
    train_arr, val_arr, test_arr = get_array_data(decom_method=config.decom_method, dim=2)
    lgb_train = lgb.Dataset(train_arr[:, :-1], train_arr[:, -1])
    lgb_val = lgb.Dataset(val_arr[:, :-1], val_arr[:, -1], reference=lgb_train)

    params = {
        'task': 'train',
        'boosting_type': 'gbdt',  # 设置提升类型
        'objective': 'regression',  # 目标函数
        'metric': {'mse'},  # 评估函数
        # 'num_leaves': 31,  # 叶子节点数
        # 'learning_rate': 0.05,  # 学习速率
        # 'feature_fraction': 0.9,  # 建树的特征选择比例
        # 'bagging_fraction': 0.8,  # 建树的样本采样比例
        # 'bagging_freq': 5,  # k 意味着每 k 次迭代执行bagging
        # 'force_col_wise': True,
        'verbosity': -1  # <0 显示致命的, =0 显示错误 (警告), >0 显示信息
    }

    gbm = lgb.train(params=params, train_set=lgb_train, valid_sets=[lgb_val],
                    callbacks=[lgb.callback.early_stopping(stopping_rounds=5, verbose=False)])
    gbm.save_model(os.path.join(config.mdoel_saving_dir, f'{config.decom_method}-LightGBM.txt'))
    y_pred, y_true = inverse_norm(gbm.predict(test_arr[:, :-1])), inverse_norm(test_arr[:, -1])
    plt.plot(range(100), y_true[200:300])
    plt.plot(range(100), y_pred[200:300])
    plt.show()
    return mean_absolute_percentage_error(y_true, y_pred)


pl.seed_everything(config.seed, workers=True)
print(f"{config.decom_method}: MAPE of GRU: {run_normal_GRU()}, MAPE of LGB: {run_lightgbm()}")  #

