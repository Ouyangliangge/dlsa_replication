import logging
import os
import time

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from utils import import_string
from matplotlib.dates import DateFormatter, MonthLocator, YearLocator, DayLocator

torch.set_default_dtype(torch.float)
torch.autograd.set_detect_anomaly(True)
torch.backends.cudnn.deterministic = False


# torch.backends.cudnn.benchmark = True
# torch.use_deterministic_algorithms()


def train(model,
          preprocess,
          data_train,
          data_dev=None,
          excess_returns_data=None,
          log_dev_progress=True, log_dev_progress_freq=50, log_plot_freq=50,
          num_epochs=100, lr=0.001, batchsize=50,  # 原为200
          optimizer_name="Adam", optimizer_opts={"lr": 0.0001},  # 原为0.001
          early_stopping=False, early_stopping_max_trials=5, lr_decay=0.5,
          residual_weights_train=None, residual_weights_dev=None,
          save_params=True, output_path=None, model_tag='',
          lookback=30,
          trans_cost=0, hold_cost=0,
          parallelize=True, device=None, device_ids=[0, 1, 2, 3, 4, 5, 6, 7],  # must use device='cuda' to parallelize
          force_retrain=True,
          objective="sharpe", vari=None):
    if output_path is None: output_path = model.logdir
    if device is None: device = model.device
    logging.info(f"train(): data_train.shape {data_train.shape}")

    # preprocess data
    # assets_to_trade chooses assets which have at least `lookback` non-missing observations in the training period
    # this does not induce lookahead bias because idxs_selected is backward-looking and
    # will only select assets with at least `lookback` non-missing obs
    assets_to_trade = np.count_nonzero(data_train, axis=0) >= lookback
    logging.info(f"train(): assets_to_trade.shape {assets_to_trade.shape}")
    data_train = data_train[:, assets_to_trade]
    excess_returns_data = excess_returns_data[:, assets_to_trade]
    if residual_weights_train is not None:
        residual_weights_train = residual_weights_train[:, assets_to_trade]
    T, N = data_train.shape
    logging.info(f"train(): T {T} N {N}")
    windows, idxs_selected = preprocess(data_train, lookback)
    logging.info(f"train(): windows.shape {windows.shape} idxs_selected.shape {idxs_selected.shape}")

    # start to train
    # 设置训练环境
    if parallelize:  # 并行化
        model = nn.DataParallel(model, device_ids=device_ids)
    model.to(device)  # 设置训练设备
    model.train()  # 设置为训练模式
    optimizer_func = import_string(f"torch.optim.{optimizer_name}")  # 创建优化器
    optimizer = optimizer_func(model.parameters(), **optimizer_opts)

    min_dev_loss = np.inf
    patience = 0
    trial = 0
    # 检查是否需要重新训练
    # 检查预训练checkpoint
    already_trained = False
    checkpoint_fname = f'{vari}Checkpoint-{model.module.random_seed if parallelize else model.random_seed}_seed_' + model_tag + '.tar'
    if os.path.isfile(os.path.join(output_path, 'Checkpoint', checkpoint_fname)) and not force_retrain:  # 文件存在且不需要重新训练
        already_trained = True
        checkpoint = torch.load(os.path.join(output_path,'Checkpoint',  checkpoint_fname), map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.train()
        logging.info('Already trained!')

    # train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False) #数据加载器

    begin_time = time.time()  # 计算整个训练过程的耗时。
    for epoch in range(num_epochs):
        if (epoch % 20 == 0):
            print(f"running epoch {epoch}/{num_epochs}")
        if epoch == num_epochs:
            print(f"running epoch {epoch}/{num_epochs}")
        rets_full = np.zeros(T - lookback)
        excess_rets_full = np.zeros(T - lookback)
        short_proportion = np.zeros(T - lookback)
        turnover = np.zeros(T - lookback)

        # break input data up into batches of size `batchsize` and train over them
        for i in range(int((T - lookback) / batchsize) + 1):  # 循环处理数据块
            # 创建权重
            weights = 2 * torch.rand((min(batchsize * (i + 1), T - lookback) - batchsize * i, N),
                                     device=device) - 1  # 随机初始化-1~1的权重
            weights = weights / (weights.sum() + 0.00000000001)  # 归一化处理，使每一行权重总和为1
            if epoch == 0 and i == 0:
                logging.info(f"epoch {epoch} batch {i} weights.shape {weights.shape}")
            else:
                logging.debug(f"epoch {epoch} batch {i} weights.shape {weights.shape}")
            logging.debug("stats: " + \
                          f"idxs_selected.shape {idxs_selected.shape}, " + \
                          f"filtered for batch {i} idxs_selected.shape {idxs_selected[batchsize * i:min(batchsize * (i + 1), T - lookback), :].shape}, " + \
                          f"weights.shape {weights.shape}, " + \
                          f"batch period len {min(batchsize * (i + 1), T) - batchsize * i}"
                          )
            # 模型权重更新
            # all stocks selected below therefore no filtering here.
            idxs_batch_i = idxs_selected[batchsize * i:min(batchsize * (i + 1), T - lookback),
                           :]  # idxs of valid residuals to trade in batch i
            input_data_batch_i = windows[batchsize * i:min(batchsize * (i + 1), T - lookback)][idxs_batch_i]
            logging.debug(f"epoch {epoch} batch {i} input_data_batch_i.shape {input_data_batch_i.shape}")
            weights[idxs_batch_i] = model(torch.tensor(input_data_batch_i, device=device))  # 前向传播更新权重
            if residual_weights_train is None:
                abs_sum = torch.sum(torch.abs(weights), axis=1, keepdim=True)  # 计算每一行权重绝对值之和
            else:  # residual_weights_train is TxN1xN2 (multiplied by returns on the right gives residuals)
                assert (weights.shape == residual_weights_train[
                                         lookback + batchsize * i:min(lookback + batchsize * (i + 1), T), :,
                                         0].shape)  # 检查形状一致
                T1, N1 = weights.shape  # weights is T1xN1
                weights2 = torch.bmm(weights.reshape(T1, 1, N1), \
                                     torch.tensor(residual_weights_train[
                                                  lookback + batchsize * i:min(lookback + batchsize * (i + 1), T)],
                                                  device=device)).squeeze()  # will be T1xN2: weights2 is in underlying asset space
                if epoch == 0 and i == 0:
                    logging.info(f"epoch {epoch} batch {i} weights2.shape {weights2.shape}")
                else:
                    logging.debug(f"epoch {epoch} batch {i} weights2.shape {weights2.shape}")
                abs_sum = torch.sum(torch.abs(weights2), axis=1, keepdim=True)
                try:
                    weights2 = weights2 / abs_sum  # 归一化每一行
                except:
                    weights2 = weights2 / (abs_sum + 1e-8)
            try:weights = weights/abs_sum
            except:weights = weights/(abs_sum+1e-8)
            # normalize wght above and calcuate returns below
            rets_train = torch.sum(
                weights * torch.tensor(data_train[lookback + batchsize * i:min(lookback + batchsize * (i + 1), T), :],
                                       device=device), axis=1)
            try:
                excess_rets_train = torch.sum(weights * torch.tensor(
                    excess_returns_data[lookback + batchsize * i:min(lookback + batchsize * (i + 1), T), :],
                    device=device), axis=1)
            except:
                print("Here")

            # 计算投资组合的总收益率，同时考虑交易成本和持有成本
            if residual_weights_train is None:
                rets_train = rets_train \
                    - trans_cost * torch.cat(
                    (torch.zeros(1, device=device),
                     torch.sum(torch.abs(weights[1:] - weights[:-1]), axis=1))) \
                    - hold_cost * torch.sum(torch.abs(torch.min(weights, torch.zeros(1, device=device))),axis=1)
                excess_rets_train = excess_rets_train \
                    - trans_cost * torch.cat(
                    (torch.zeros(1, device=device),
                     torch.sum(torch.abs(weights[1:] - weights[:-1]), axis=1))) \
                    - hold_cost * torch.sum(torch.abs(torch.min(weights, torch.zeros(1, device=device))), axis=1)
            else:
                rets_train = rets_train \
                    - trans_cost * torch.cat(
                    (torch.zeros(1, device=device),
                     torch.sum(torch.abs(weights2[1:] - weights2[:-1]), axis=1))) \
                    - hold_cost * torch.sum(torch.abs(torch.min(weights2, torch.zeros(1, device=device))),axis=1)
                excess_rets_train = excess_rets_train \
                    - trans_cost * torch.cat(
                    (torch.zeros(1, device=device),
                     torch.sum(torch.abs(weights2[1:] - weights2[:-1]), axis=1))) \
                    - hold_cost * torch.sum(torch.abs(torch.min(weights2, torch.zeros(1, device=device))), axis=1)

            mean_ret = torch.mean(rets_train) #平均收益率
            std = torch.std(rets_train) #标准差
            mean_excess_ret = torch.mean(excess_rets_train) #平均超额收益率
            std_excess = torch.std(excess_rets_train) #超额收益率标准差

            if objective == "sharpe":
                loss = -mean_excess_ret/std_excess
            elif objective == "meanvar":
                loss = -mean_excess_ret*252 + std_excess*15.9
            elif objective == "sqrtMeanSharpe":
                loss = -torch.sign(mean_excess_ret) * torch.sqrt(torch.abs(mean_excess_ret)) / (torch.std(mean_excess_ret)+0.000000000001)
            else:
                raise Exception(f"Invalid objective loss {objective}")

            if not already_trained and ((parallelize and model.module.is_trainable) or (not parallelize and model.is_trainable)):
                # calcuate weight here that maximize sharpe
                optimizer.zero_grad() #优化器梯度清零
                loss.backward() #计算损失函数梯度
                optimizer.step() #根据梯度更新模型的权重，以最小化损失函数
            #/提取权重
            if residual_weights_train is None:
                weights = weights.detach().cpu().numpy()
            else:
                weights = weights2.detach().cpu().numpy()

            rets_full[batchsize * i:min(batchsize * (i + 1), T - lookback)] = rets_train.detach().cpu().numpy()
            excess_rets_full[batchsize * i:min(batchsize * (i + 1), T - lookback)] = excess_rets_train.detach().cpu().numpy()
            turnover[batchsize * i:(min(batchsize * (i + 1), T - lookback) - 1)] = np.sum(np.abs(weights[1:] - weights[:-1]), axis=1)
            turnover[min(batchsize * (i + 1), T - lookback) - 1] = turnover[min(batchsize * (i + 1), T - lookback) - 2]  # just to simplify things
            short_proportion[batchsize * i:min(batchsize * (i + 1), T - lookback)] = np.sum(np.abs(np.minimum(weights, 0)), axis=1)

        if log_dev_progress and epoch % log_dev_progress_freq == 0:
            dev_loss_description = ""
            if data_dev is not None:
                rets_dev, dev_loss, dev_sharpe, dev_turnovers, dev_short_proportions, weights_dev, a2t, excess_ret = \
                    get_returns(model,
                                preprocess=preprocess,
                                objective=objective,
                                data_test=data_dev,
                                excess_returns_data=excess_returns_data,
                                device=device,
                                lookback=lookback,
                                trans_cost=trans_cost, hold_cost=hold_cost,
                                residual_weights=residual_weights_dev)

                model.train()
                dev_mean_ret = np.mean(rets_dev)
                dev_std = np.std(rets_dev)
                dev_turnover = np.mean(dev_turnovers)
                dev_short_proportion = np.mean(dev_short_proportions)
                dev_loss_description = f", dev loss {-dev_loss:0.2f}, " \
                                       f"dev Sharpe: {-dev_sharpe * np.sqrt(252):0.2f}, " \
                                       f"ret: {dev_mean_ret * 252:0.4f}, " \
                                       f"std: {dev_std * np.sqrt(252) :0.4f}, " \
                                       f"turnover: {dev_turnover:0.3f}, " \
                                       f"short proportion: {dev_short_proportion:0.3f}\n"

            full_ret = np.mean(excess_rets_full)
            full_std = np.std(excess_rets_full)
            full_sharpe = full_ret / full_std
            full_turnover = np.mean(turnover)
            full_short_proportion = np.mean(short_proportion)

            logging.info(f'Epoch: {epoch}/{num_epochs}, ' \
                         f'train Sharpe: {full_sharpe * np.sqrt(252):0.2f}, ' \
                         f'ret: {full_ret * 252:0.4f}, ' \
                         f'std: {full_std * np.sqrt(252):0.4f}, ' \
                         f'turnover: {full_turnover:0.3f}, ' \
                         f'short proportion: {full_short_proportion:0.3f} \n' \
                         '       ' \
                         f' time per epoch: {(time.time() - begin_time) / (epoch + 1):0.2f}s' \
                         + dev_loss_description)

            if early_stopping:
                if dev_loss < min_dev_loss:
                    patience = 0
                    min_dev_loss = dev_loss
                    checkpoint = {
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': loss
                    }
                    torch.save(checkpoint,
                               os.path.join(output_path,'Checkpoint',  f'{vari}Checkpoint-{model.random_seed}_seed_{model_tag}.tar'))
                else:
                    patience += 1
                    if trial == early_stopping_max_trials:
                        logging.info('Early stopping max trials reached')
                        break
                    else:  # reduce learning rate
                        trial += 1
                        logging.info('Reducing learning rate')
                        lr = optimizer.param_groups[0]['lr'] * lr_decay
                        checkpoint = torch.load(os.path.join(output_path, 'Checkpoint', \
                                                             f'{vari}Checkpoint-{model.random_seed}_seed_{model_tag}.tar'),
                                                map_location=device)
                        model.load_state_dict(checkpoint['model_state_dict'])
                        model = model.to(device)
                        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                        model.train()
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr
                        patience = 0

            # if epoch == num_epochs-1 and data_dev is not None and log_dev_progress: # or (epoch % log_plot_freq == 0)
            #     #cum_rets_train = np.cumprod(1+rets_train.detach().cpu().numpy())
            #     plt.figure()
            #     cum_rets_train = np.cumprod(1+rets_full)
            #     cum_rets_dev = np.cumprod(1+rets_dev)
            #     plt.plot(cum_rets_train,label='Train')
            #     plt.plot(cum_rets_dev, label='Dev')
            #     plt.title('Cumulative returns')
            #     plt.legend()
            #     plt.show()
            #
            #     plt.figure()
            #     plt.plot(turnover,label='Train')
            #     plt.plot(dev_turnovers, label='Dev')
            #     plt.title('Turnover')
            #     plt.legend()
            #     plt.show()
            #
            #     plt.figure()
            #     plt.plot(short_proportion,label='Train')
            #     plt.plot(dev_short_proportions, label='Dev')
            #     plt.title('Short proportion')
            #     plt.legend()
            #     plt.show()

        if already_trained: break

    if save_params and not already_trained:
        # can also save model.state_dict() directly w/o the dictionary; extension should then be .pth instead of .tar
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss
        }
        checkpoint_fname = f'Checkpoint-{model.module.random_seed if parallelize else model.random_seed}_seed_' + model_tag + '.tar'
        torch.save(checkpoint, os.path.join(output_path,'Checkpoint', checkpoint_fname))

    logging.info(
        f'Training done - Model: {model_tag}, seed: {model.module.random_seed if parallelize else model.random_seed}')
    if data_dev is not None:
        print("XXXXXXXXXXXXXXXX")
        return rets_dev, dev_turnovers, dev_short_proportions, weights_dev, a2t
    else:
        return rets_full, turnover, short_proportion, weights, assets_to_trade, model


def get_returns(model,
                preprocess,
                objective,
                data_test,
                excess_returns_data,
                lookback=30,
                trans_cost=0,
                hold_cost=0,
                residual_weights=None,
                load_params=False,
                paths_checkpoints=[None],
                device=None,
                parallelize=False,
                device_ids=[0, 1, 2, 3, 4, 5, 6, 7], ):
    if device is None: device = model.device
    if parallelize: model = nn.DataParallel(model, device_ids=device_ids).to(device)

    # restrict to assets which have at least `lookback` non-missing observations in the training period
    assets_to_trade = np.count_nonzero(data_test, axis=0) >= lookback
    logging.debug(f"get_returns(): assets_to_trade.shape {assets_to_trade.shape}")
    data_test = data_test[:, assets_to_trade]
    excess_returns_data = excess_returns_data[:, assets_to_trade]
    T, N = data_test.shape
    windows, idxs_selected = preprocess(data_test, lookback)

    rets_test = torch.zeros(T - lookback)
    excess_ret = torch.zeros(T - lookback)
    model.eval()

    with torch.no_grad():
        weights = torch.zeros((T - lookback, N), device=device)
        for i in range(len(paths_checkpoints)):  # This ensembles if many checkpoints are given
            if load_params:
                checkpoint = torch.load(paths_checkpoints[i], map_location=device)
                model.load_state_dict(checkpoint['model_state_dict'])
                model.to(device)
            weights[idxs_selected] += model(torch.tensor(windows[idxs_selected], device=device))
        weights /= len(paths_checkpoints)
        if residual_weights is None:
            abs_sum = torch.sum(torch.abs(weights), axis=1, keepdim=True)
            logging.debug(f"get_returns(): weights abs_sum {abs_sum / len(weights)}")
        else:
            residual_weights = residual_weights[:, assets_to_trade]
            assert (weights.shape == residual_weights[lookback:T, :, 0].shape)
            T1, N1 = weights.shape
            weights2 = torch.bmm(weights.reshape(T1, 1, N1),
                                 torch.tensor(residual_weights[lookback:T], device=device)).squeeze()
            abs_sum = torch.sum(torch.abs(weights2), axis=1, keepdim=True)
            logging.debug(f"get_returns(): weights2 abs_sum {abs_sum / len(weights2)}")
            try:
                weights2 = weights2 / abs_sum
            except:
                weights2 = weights2 / (abs_sum + 1e-8)
        try:
            weights = weights / abs_sum
        except:
            weights = weights / (abs_sum + 1e-8)
        rets_test = torch.sum(weights * torch.tensor(data_test[lookback:T, :], device=device), axis=1)
        excess_ret = torch.sum(weights * torch.tensor(excess_returns_data[lookback:T, :], device=device), axis=1)
        if residual_weights is not None:
            weights = weights2
        turnover = torch.cat((torch.zeros(1, device=device), torch.sum(torch.abs(weights[1:] - weights[:-1]), axis=1)))
        short_proportion = torch.sum(torch.abs(torch.min(weights, torch.zeros(1, device=device))), axis=1)
        rets_test = rets_test - trans_cost * turnover - hold_cost * short_proportion
        turnover[0] = torch.mean(turnover[1:])
        mean = torch.mean(rets_test)
        std = torch.std(rets_test)
        sharpe = -mean / std
        loss = None
        if objective == "sharpe":
            loss = sharpe
        elif objective == "meanvar":
            loss = -mean * 252 + std * 15.9
        elif objective == "sqrtMeanSharpe":
            loss = -torch.sign(mean) * torch.sqrt(torch.abs(mean)) / std
        else:
            raise Exception(f"Invalid objective loss {objective}")
    return (rets_test.cpu().numpy(), loss, sharpe, turnover.cpu().numpy(), short_proportion.cpu().numpy(),
             weights.cpu().numpy(),assets_to_trade,excess_ret)


def test(Data,
         daily_dates,
         model,
         preprocess,
         config,
         excess_returns_data,
         residual_weights=None,
         log_dev_progress_freq=50, log_plot_freq=199,
         num_epochs=100, lr=0.001, batchsize=50,#150
         early_stopping=False,
         save_params=True,
         device='cuda',
         output_path=os.path.join(os.getcwd(), 'results'), model_tag='Unknown',
         lookback=30, retrain_freq=250, length_training=1000, rolling_retrain=True,
         parallelize=True,
         device_ids=[0, 1, 2, 3, 4, 5, 6, 7],
         trans_cost=0, hold_cost=0,
         force_retrain=False,
         objective="sharpe", vari=None):
    # chooses assets which have at least #lookback non-missing observations in the training period
    assets_to_trade = np.count_nonzero(Data, axis=0) >= lookback
    # logging.info(f"test(): assets_to_trade.shape {assets_to_trade.shape}")
    Data = Data[:, assets_to_trade]
    excess_returns_data = excess_returns_data[:, assets_to_trade]
    T, N = Data.shape
    returns = np.zeros(T - length_training)
    excess_returns = np.zeros(T - length_training)
    turnovers = np.zeros(T - length_training)
    short_proportions = np.zeros(T - length_training)
    all_weights = np.zeros((T - length_training, len(assets_to_trade)))

    # run train/test over dataset
    for t in range(int((T - length_training) / retrain_freq) + 1):
        print(f"running test in {t+1}/{int((T - length_training) / retrain_freq)+1} this many times test will play")
        # logging.info(f'AT SUBPERIOD {t}/{int((T - length_training) / retrain_freq) + 1}')
        # logging.info(f"{Data[initialTrain:length_training+(t)*retrain_freq].shape} {Data[length_training+t*retrain_freq:min(length_training+(t+1)*retrain_freq,T)].shape}")
        data_train_t = Data[t * retrain_freq:length_training + t * retrain_freq]
        data_test_t = Data[length_training + t * retrain_freq - lookback:min(length_training + (t + 1) * retrain_freq, T)]
        excess_returns_data_train_t = excess_returns_data[t * retrain_freq:length_training + t * retrain_freq]
        excess_returns_data_test_t = excess_returns_data[length_training + t * retrain_freq - lookback:min(length_training + (t + 1) * retrain_freq, T)]
        residual_weights_train_t = None if residual_weights is None \
            else residual_weights[t * retrain_freq:length_training + t * retrain_freq]
        residual_weights_test_t = None if residual_weights is None \
            else residual_weights[
                 length_training + t * retrain_freq - lookback:min(length_training + (t + 1) * retrain_freq, T)]
        model_tag_t = model_tag + f'__subperiod{t}'

        if rolling_retrain or t == 0:
            #to train the model
            model_t = model
            print(f"{t+1}/{int((T-length_training)/retrain_freq)+1} the value of t started")
            rets_t, turns_t, shorts_t, weights_t, a2t, model_t = train(model_t,
                                                              preprocess=preprocess,
                                                              data_train=data_train_t,
                                                              # data_dev=data_test_t,
                                                              # dev dataset isn't used as we don't do any validation tuning, so test dataset goes here for progress reporting
                                                              excess_returns_data=excess_returns_data_train_t,
                                                              residual_weights_train=residual_weights_train_t,
                                                              # residual_weights_dev=residual_weights_test_t,
                                                              # dev dataset isn't used as we don't do any validation tuning, so test dataset goes here for progress reporting
                                                              log_dev_progress_freq=log_dev_progress_freq,
                                                              num_epochs=num_epochs,
                                                              force_retrain=force_retrain,
                                                              optimizer_name=config['optimizer_name'],
                                                              optimizer_opts=config['optimizer_opts'],
                                                              early_stopping=early_stopping,
                                                              save_params=save_params,
                                                              output_path=output_path,
                                                              model_tag=model_tag_t,
                                                              device=device,
                                                              lookback=lookback,
                                                              log_plot_freq=log_plot_freq,
                                                              parallelize=parallelize,
                                                              device_ids=device_ids,
                                                              batchsize=batchsize,
                                                              trans_cost=trans_cost,
                                                              hold_cost=hold_cost,
                                                              objective=objective, vari=vari)
            rolling_retrain = False
        if rolling_retrain == False :
            rets_t, _, _, turns_t, shorts_t, weights_t, a2t,excess_ret = get_returns(model_t,
                                                                          preprocess=preprocess,
                                                                          objective=objective,
                                                                          data_test=data_test_t,
                                                                          excess_returns_data=excess_returns_data_test_t,
                                                                          residual_weights=residual_weights_test_t,
                                                                          device=device,
                                                                          lookback=lookback,
                                                                          trans_cost=trans_cost,
                                                                          hold_cost=hold_cost, )
            rolling_retrain = True

        returns[t * retrain_freq:min((t + 1) * retrain_freq, T - length_training)] = rets_t
        excess_returns[t * retrain_freq:min((t + 1) * retrain_freq, T - length_training)] = excess_ret
        turnovers[t * retrain_freq:min((t + 1) * retrain_freq, T - length_training)] = turns_t
        short_proportions[t * retrain_freq:min((t + 1) * retrain_freq, T - length_training)] = shorts_t
        if residual_weights is None:
            w = np.zeros((min((t + 1) * retrain_freq, T - length_training) - t * retrain_freq, len(a2t)))
            w[:, a2t] = weights_t
        else:
            w = weights_t
        all_weights[t * retrain_freq:min((t + 1) * retrain_freq, T - length_training), assets_to_trade] = w
        if 'cpu' not in device:
            with torch.cuda.device(device):
                torch.cuda.empty_cache()

    print(f'TRAIN/TEST COMPLETE')
    cumRets = np.cumprod(1 + returns)
    plt.figure()
    plt.plot_date(daily_dates[-len(cumRets):], cumRets, marker='None',linestyle='solid')
    ax = plt.gca()
    ax.xaxis.set_major_locator(YearLocator(2))  # Show ticks every 2 years
    ax.xaxis.set_minor_locator(DayLocator(interval=30))  # Place data points every 30 days
    ax.xaxis.set_major_formatter(DateFormatter('%Y'))  # Format tick labels as YYYY
    plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
    plt.title("Cumulative Returns Over Time")  # Adding a title
    plt.xlabel("Dates")  # Adding an x-label
    plt.ylabel("Cumulative Returns")  # Adding a y-label
    plt.tight_layout()  # Improve spacing and prevent overlap
    plt.savefig(os.path.join(output_path, 'cumulative_returns', model_tag + f"{vari}_cumulative-returns.png"), bbox_inches='tight')
    # plt.show()

    plt.figure()
    plt.plot_date(daily_dates[-len(cumRets):], turnovers,marker='None', linestyle='solid')
    ax = plt.gca()
    ax.xaxis.set_major_locator(YearLocator(2))  # Show ticks every 2 years
    ax.xaxis.set_minor_locator(DayLocator(interval=30))  # Place data points every 30 days
    ax.xaxis.set_major_formatter(DateFormatter('%Y'))  # Format tick labels as YYYY
    plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
    plt.title("Turnover Over Time")  # Adding a title
    plt.xlabel("Dates")  # Adding an x-label
    plt.ylabel("Turnover")  # Adding a y-label
    plt.tight_layout()  # Improve spacing and prevent overlap
    plt.savefig(os.path.join(output_path, 'turnovers', model_tag + f"{vari}_turnover.png"))
    # plt.show()

    plt.figure()
    plt.plot_date(daily_dates[-len(cumRets):], short_proportions,marker='None', linestyle='solid')
    ax = plt.gca()
    ax.xaxis.set_major_locator(YearLocator(2))  # Show ticks every 2 years
    ax.xaxis.set_minor_locator(DayLocator(interval=30))  # Place data points every 30 days
    ax.xaxis.set_major_formatter(DateFormatter('%Y'))  # Format tick labels as YYYY
    plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
    plt.title("Short Proportion Over Time")  # Adding a title
    plt.xlabel("Dates")  # Adding an x-label
    plt.ylabel("Short")  # Adding a y-label
    plt.tight_layout()  # Improve spacing and prevent overlap
    plt.savefig(os.path.join(output_path, 'short_proportions', model_tag + f"{vari}_short-proportion.png"))
    # plt.show()

    # cumExRets = np.cumprod(1 + excess_returns)
    # plt.figure()
    # plt.plot_date(daily_dates[-len(cumRets):], cumExRets,marker='None', linestyle='solid')
    # ax = plt.gca()
    # ax.xaxis.set_major_locator(YearLocator(2))  # Show ticks every 2 years
    # ax.xaxis.set_minor_locator(DayLocator(interval=30))  # Place data points every 30 days
    # ax.xaxis.set_major_formatter(DateFormatter('%Y'))  # Format tick labels as YYYY
    # plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
    # plt.title("Excess_Returns Over Time")  # Adding a title
    # plt.xlabel("Dates")  # Adding an x-label
    # plt.ylabel("Excess Returns")  # Adding a y-label
    # plt.tight_layout()  # Improve spacing and prevent overlap
    # plt.savefig(os.path.join(output_path, 'exc_cum_returns', model_tag + f"{vari}_excess_cumulative-returns.png"), bbox_inches='tight')
    # plt.show()

    np.save(os.path.join(output_path, 'Unknown', f'{vari}WeightsComplete_' + model_tag + '.npy'), all_weights)
    np.save(os.path.join(output_path, 'Unknown', f'{vari}cumalphaRets_' + model_tag + '.npy'), cumRets)
    np.save(os.path.join(output_path, 'Unknown', f'{vari}alphaReturns' + model_tag + '.npy'), returns)
    np.save(os.path.join(output_path, 'Unknown', f'{vari}short_' + model_tag + '.npy'), short_proportions)
    np.save(os.path.join(output_path, 'Unknown', f'{vari}turnover' + model_tag + '.npy'), turnovers)
    # np.save(os.path.join(output_path, 'Unknown', f'{vari}ExcessReturns' + model_tag + '.npy'), excess_returns)
    # np.save(os.path.join(output_path, 'Unknown', f'{vari}CumulativeExcessReturns' + model_tag + '.npy'), cumExRets)

    full_ret = np.mean(returns)
    full_std = np.std(returns)
    full_sharpe = full_ret / full_std

    print(f'{model_tag}_{vari} training completed')
    return returns, full_sharpe, full_ret, full_std, turnovers, short_proportions


def estimate(Data,
             daily_dates,
             model,
             preprocess,
             config,
             residual_weights=None,
             log_dev_progress_freq=50, log_plot_freq=199,
             num_epochs=100, lr=0.001, batchsize=150,
             early_stopping=False,
             save_params=True,
             device='cuda',
             output_path=os.path.join(os.getcwd(), 'results'), model_tag='Unknown',
             lookback=30, length_training=1000, test_size=125,
             parallelize=True,
             device_ids=[0, 1, 2, 3, 4, 5, 6, 7],
             trans_cost=0, hold_cost=0,
             force_retrain=True,
             objective="sharpe",
             estimate_start_idx=0,vari=None ):
    # chooses assets which have at least #lookback non-missing observations in the training period
    assets_to_trade = np.count_nonzero(Data, axis=0) >= lookback
    Data = Data[:, assets_to_trade]
    T, N = Data.shape
    returns = np.zeros(length_training)
    turnovers = np.zeros(length_training)
    short_proportions = np.zeros(length_training)
    all_weights = np.zeros((length_training, len(assets_to_trade)))

    # estimate over dataset
    logging.info(f"ESTIMATING {estimate_start_idx}:{min(estimate_start_idx + length_training, T)}")
    logging.info(
        f"TESTING {estimate_start_idx + length_training - lookback}:{min(estimate_start_idx + length_training + test_size, T)}")
    data_train = Data[estimate_start_idx:min(estimate_start_idx + length_training, T)]
    data_dev = Data[
               estimate_start_idx + length_training - lookback:min(estimate_start_idx + length_training + test_size, T)]
    residual_weights_train = None if residual_weights is None \
        else residual_weights[estimate_start_idx:min(estimate_start_idx + length_training, T)]
    residual_weights_dev = None if residual_weights is None \
        else residual_weights[
             estimate_start_idx + length_training - lookback:min(estimate_start_idx + length_training + test_size, T)]
    del residual_weights
    del Data
    model_tag = model_tag + f'__estimation{estimate_start_idx}-{length_training}-{test_size}'

    model1 = model(logdir=output_path, **config['model'])
    rets, turns, shorts, weights = train(model1,
                                         preprocess=preprocess,
                                         data_train=data_train,
                                         data_dev=data_dev,
                                         residual_weights_train=residual_weights_train,
                                         residual_weights_dev=residual_weights_dev,
                                         log_dev_progress_freq=log_dev_progress_freq,
                                         num_epochs=num_epochs,
                                         force_retrain=force_retrain,
                                         lr=lr,
                                         early_stopping=early_stopping,
                                         save_params=save_params,
                                         output_path=output_path,
                                         model_tag=model_tag,
                                         device=device,
                                         lookback=lookback,
                                         log_plot_freq=log_plot_freq,
                                         parallelize=parallelize,
                                         device_ids=device_ids,
                                         batchsize=batchsize,
                                         trans_cost=trans_cost,
                                         hold_cost=hold_cost,
                                         objective=objective, vari=None)

    returns = rets
    turnovers = turns
    short_proportions = shorts
    all_weights = weights
    if 'cpu' not in device:
        with torch.cuda.device(device):
            torch.cuda.empty_cache()

    logging.info(f'ESTIMATION COMPLETE')

    np.save(os.path.join(output_path,'Weights',  'WeightsComplete_' + model_tag + '.npy'), all_weights)

    full_ret = np.mean(returns)
    full_std = np.std(returns)
    full_sharpe = full_ret / full_std
    logging.info(f"==> Sharpe: {full_sharpe * np.sqrt(252) :.2f}, " \
                 f"ret: {full_ret * 252 :.4f}, " \
                 f"std: {full_std * np.sqrt(252) :.4f}, " \
                 f"turnover: {np.mean(turnovers) :.4f}, " \
                 f"short_proportion: {np.mean(short_proportions) :.4f}")

    return returns, full_sharpe, full_ret, full_std, turnovers, short_proportions
