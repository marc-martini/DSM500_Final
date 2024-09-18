from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models import Informer, Transformer, DLinear, NLinear, PatchTST, FEDformer, RevIN_Linear, PatchFED,  RevIN_PatchTST, Linear_PatchTST, PatchFED_full
from utils.tools import EarlyStopping, adjust_learning_rate, visual, test_params_flop, torch_summarize
from utils.metrics import metric

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler 

import os
import time

import warnings
import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings('ignore')

class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)
        self.full_start_time = args.full_start_time
        self.data_name = args.data_name
    
    
    # build the model for the experiment based on the imported model dictionary 
    def _build_model(self):
        model_dict = {
            'Informer': Informer,
            'PatchTST': PatchTST,
            'DLinear': DLinear,
            'NLinear': NLinear,
            'FEDformer': FEDformer,
            'Linear_PatchTST': Linear_PatchTST,
            'PatchFED' : PatchFED,
            'RevIN_PatchTST' : RevIN_PatchTST,
            'RevIN_Linear' : RevIN_Linear
        }
        self.args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        model = model_dict[self.args.model].Model(self.args).float()
        model = model.to(self.device)
        
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model
        
        # check for gpu execution 
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    # get the data handlers from data factory
    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    # define model optimiser 
    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    # define trianing metric criteira/ loss metric 
    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    # validation function
    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        # set model mode to validation 
        self.model.eval()
        with torch.no_grad():
            # get bacth data and enumerate over 
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                # assign data to gpu device 
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'Linear' in self.args.model or 'TST' in self.args.model:
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if 'Linear' in self.args.model or 'TST' in self.args.model:
                        outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)

                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        # set train start time and time vari
        train_start_time = time.time()
        self.train_time = 0

        # load the data loaders for train, vali and test 
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        # set the path for the model to be saved
        path = os.path.join('./results/', self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)
                

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()
            
        scheduler = lr_scheduler.OneCycleLR(optimizer = model_optim,
                                            steps_per_epoch = train_steps,
                                            pct_start = self.args.pct_start,
                                            epochs = self.args.train_epochs,
                                            max_lr = self.args.learning_rate)
        
        folder_path = os.path.join('./results/', setting)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        file_name = folder_path + '/Model training and test.txt'
        train_f = open(file_name, 'a')
        train_f.write('Training outputs:\n')
        train_f.write('\n')
        train_f.write(setting + "  \n")
        train_f.write('\n')
        train_f.close()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)

                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'Linear' in self.args.model or 'TST' in self.args.model:
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:] 
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    if 'Linear' in self.args.model or 'TST' in self.args.model:
                            outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                            #outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_y)
                    # print(outputs.shape,batch_y.shape)
                    keep_outputs = outputs
                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())                   

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    train_f = open(file_name, 'a')
                    train_f.write("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    train_f.write('\n')
                    speed = (time.time() - train_start_time) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    train_f = open(file_name, 'a')
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    train_f.write('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    train_f.write('\n')
                    train_f.close()
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()
                    
                if self.args.lradj == 'TST':
                    adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args, printout=False)
                    scheduler.step()

            train_f = open(file_name, 'a')
            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_f.write("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_f.write('\n')
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            train_f.write("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            train_f.write('\n')
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                train_f.write("Early stopping")
                train_f.write('\n')
                break

            if self.args.lradj != 'TST':
                adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args)
            else:
                print('Updating learning rate to {}'.format(scheduler.get_last_lr()[0]))
                train_f.write('Updating learning rate to {}'.format(scheduler.get_last_lr()[0]))
                train_f.write('\n')

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        self.size_model = os.path.getsize(('./results/checkpoints/' + setting + '/checkpoint.pth'))

        # end train time and update vari
        self.training_time = time.time() - train_start_time
        
        train_f.close()

        return self.model

    def test(self, setting, test=0):

        # test time start
        test_time_start = time.time()
        self.test_time = 0

        folder_path = os.path.join('./results/', setting)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        test_file_name = folder_path + '/Model training and test.txt'
        test_f = open(test_file_name, 'a')
        test_f.write('Testing outputs:\n')
        
        test_data, test_loader = self._get_data(flag='test')

        test_time_now = time.time()
        
        if test:
            print('loading model')
            test_f.write('loading model')
            test_f.write('\n')
            self.model.load_state_dict(torch.load(os.path.join('./results/checkpoints/' + setting, 'checkpoint.pth')))
        
        model_parameters = sum(p.numel() for p in self.model.parameters())

        preds = []
        trues = []
        inputx = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'Linear' in self.args.model or 'TST' in self.args.model:
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if 'Linear' in self.args.model or 'TST' in self.args.model:
                            outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]

                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                pred = outputs  # outputs.detach().cpu().numpy()  # .squeeze()
                true = batch_y  # batch_y.detach().cpu().numpy()  # .squeeze()

                preds.append(pred)
                trues.append(true)
                inputx.append(batch_x.detach().cpu().numpy())
                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))


        
        summ_file_name = folder_path + '/model_summary.txt'
        summ_f = open(summ_file_name, 'a')
        summ_f.write(setting + "  \n")
        summ_f.write('\n')
        summ_f.write(torch_summarize(self.model))
        summ_f.write('\n')
        summ_f.write('\n')
        summ_f.write('Model size:{}bytes,   {}Kb'.format(self.size_model, self.size_model/1000))
        summ_f.write('\n')
        summ_f.write('Model parameters: {}'.format(model_parameters))
        summ_f.write('\n')
        summ_f.write('\n')
        summ_f.close()
        
        if self.args.test_flop:
            test_params_flop((batch_x.shape[1],batch_x.shape[2]))
            exit()
        preds = np.array(preds)
        trues = np.array(trues)
        inputx = np.array(inputx)

        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        inputx = inputx.reshape(-1, inputx.shape[-2], inputx.shape[-1])

        # result save
        results_path = './results/' + '/'
        results_file = results_path + 'results.txt'
        #if not os.path.exists(folder_path):
        #    os.makedirs(folder_path)
        #print(folder_path)
        mae, mse, rmse, mape, mspe, rse, corr = metric(preds, trues)
        print('mse:{}, mae:{}, rse:{}'.format(mse, mae, rse))
        test_f = open(test_file_name, 'a')
        test_f.write('mse:{}, mae:{}, rse:{}'.format(mse, mae, rse))


        results_f = open(results_file, 'a')
        results_f.write(setting + "  \n")
        results_f.write('mse:{}, mae:{}, rse:{}'.format(mse, mae, rse))
        results_f.write('\n')
        results_f.write('Training time:{}s, {} min, {} hours'.format(self.training_time, (self.training_time/60), ((self.training_time/60)/60)))
        results_f.write('\n')
        self.test_time = time.time() - test_time_start
        results_f.write('Testing time:{}s, {} min, {} hours'.format((self.test_time), (self.test_time/60), ((self.test_time/60)/60)))
        results_f.write('\n')
        results_f.write('Full time:{}s, {} min, {} hours'.format((time.time() - self.full_start_time), ((time.time() - self.full_start_time)/60), (((time.time() - self.full_start_time)/60)/60)))
        results_f.write('\n')
        results_f.write('Model size:{}bytes,   {}Kb'.format(self.size_model, self.size_model/1000))
        results_f.write('\n')
        results_f.write('\n')
        results_f.write('\n')
        results_f.close()

        #np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe,rse, corr]))
        #np.save(folder_path + '/pred.npy', preds)
        #np.save(folder_path + '/true.npy', trues)
        #np.save(folder_path + '/x.npy', inputx)
        return

    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')

        if load:
            #path = os.path.join(self.args.checkpoints, setting)
            #best_model_path = path + '/' + 'checkpoint.pth'
            #self.model.load_state_dict(torch.load(best_model_path))
            self.model.load_state_dict(torch.load(os.path.join('./results/checkpoints/' + setting + '/checkpoint.pth')))

        preds = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros([batch_y.shape[0], self.args.pred_len, batch_y.shape[2]]).float().to(batch_y.device)
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if 'Linear' in self.args.model or 'TST' in self.args.model:
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if 'Linear' in self.args.model or 'TST' in self.args.model:
                        outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                pred = outputs.detach().cpu().numpy()  # .squeeze()
                preds.append(pred)

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + 'real_prediction.npy', preds)

        return

