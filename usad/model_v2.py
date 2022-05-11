import torch
import numpy as np
from torch import nn
from torch import optim

from typing import Sequence
from usad.data import SlidingWindowDataset, SlidingWindowDataLoader

import time

from scipy.stats import pearsonr
import scipy
import copy
from torch.distributions import Normal
import math


class Encoder(nn.Module):

    def __init__(self, input_dims: int, z_dims: int, nn_size: Sequence[int] = None):
        super().__init__()
        if not nn_size:
            nn_size = (input_dims // 2, input_dims // 4)

        layers = []
        last_size = input_dims
        for cur_size in nn_size:
            layers.append(nn.Linear(last_size, cur_size))
            layers.append(nn.ReLU())
            last_size = cur_size
        layers.append(nn.Linear(last_size, z_dims))
        layers.append(nn.ReLU())
        self._net = nn.Sequential(*layers)

    def forward(self, w: torch.Tensor) -> torch.Tensor:
        z = self._net(w)
        return z


class Decoder(nn.Module):

    def __init__(self, z_dims: int, input_dims: int, nn_size: Sequence[int] = None):
        super().__init__()
        if not nn_size:
            nn_size = (input_dims // 4, input_dims // 2)
        layers = []
        last_size = z_dims
        for cur_size in nn_size:
            layers.append(nn.Linear(last_size, cur_size))
            layers.append(nn.ReLU())
            last_size = cur_size
        layers.append(nn.Linear(last_size, input_dims))
        layers.append(nn.Sigmoid())
        self._net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        w = self._net(z)
        return w


class USAD:

    def __init__(self, x_dims: int, max_epochs: int = 250, batch_size: int = 128,
                 encoder_nn_size: Sequence[int] = None, decoder_nn_size: Sequence[int] = None,
                 z_dims: int = 38, window_size: int = 10, valid_step_frep: int = 200):
        self._x_dims = x_dims
        self._max_epochs = max_epochs
        self._batch_size = batch_size
        self._encoder_nn_size = encoder_nn_size
        self._decoder_nn_size = decoder_nn_size
        self._z_dims = z_dims
        self._window_size = window_size
        self._input_dims = x_dims * window_size
        self._valid_step_freq = valid_step_frep
        self._step = 0

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self._shared_encoder = Encoder(input_dims=self._input_dims, z_dims=self._z_dims)
        self._decoder_G = Decoder(z_dims=self._z_dims, input_dims=self._input_dims)
        self._decoder_D = Decoder(z_dims=self._z_dims, input_dims=self._input_dims)

        if self.device == torch.device('cuda'):
            self._shared_encoder.cuda()
            self._decoder_G.cuda()
            self._decoder_D.cuda()
            print('gpu')
        else:
            print('cpu')

        self._optimizer_G = optim.Adam(list(self._shared_encoder.parameters()) + list(self._decoder_G.parameters()))
        self._optimizer_D = optim.Adam(list(self._shared_encoder.parameters()) + list(self._decoder_D.parameters()))
    
        self.mse_left = {'AE_G': {'train': [0], 'valid': [0]},
                         'AE_D': {'train': [0], 'valid': [0]}}
        self.mse_right = {'AE_G': {'train': [0], 'valid': [0]},
                          'AE_D': {'train': [0], 'valid': [0]}}
        self.loss = {'AE_G': {'train': [0], 'valid': [0]},
                     'AE_D': {'train': [0], 'valid': [0]}}
        
    def fit(self, values, valid_portion=0.2):
        n = int(len(values) * valid_portion)
        train_values, valid_values = values[:-n], values[-n:]

        train_sliding_window = SlidingWindowDataLoader(
            SlidingWindowDataset(train_values, self._window_size),
            batch_size=self._batch_size,
            shuffle=True,
            drop_last=True
        )

        valid_sliding_window = SlidingWindowDataLoader(
            SlidingWindowDataset(valid_values, self._window_size),
            batch_size=self._batch_size,
        )

        mse = nn.MSELoss()

        total_time = 0
        for epoch in range(1, self._max_epochs+1):
            st_epoch = time.time()

            for i, w in enumerate(train_sliding_window):
                w = w.view(-1, self._input_dims)
                w = w.cuda() if self.device == torch.device('cuda') else w

                self._optimizer_G.zero_grad()
                self._optimizer_D.zero_grad()

                z = self._shared_encoder(w)
                w_G = self._decoder_G(z).cuda() if self.device == torch.device('cuda') else self._decoder_G(z)
                w_D = self._decoder_D(z).cuda() if self.device == torch.device('cuda') else self._decoder_D(z)
                w_G_D = self._decoder_D(self._shared_encoder(w_G)).cuda() if self.device == torch.device('cuda') else self._decoder_D(self._shared_encoder(w_G))

                mse_left_G = mse(w_G, w).cuda() if self.device == torch.device('cuda') else mse(w_G, w)
                mse_right_G = mse(w_G_D, w).cuda() if self.device == torch.device('cuda') else mse(w_G_D, w)
                loss_G = (1 / epoch) * mse_left_G + (1 - 1 / epoch) * mse_right_G

                self.mse_left['AE_G']['train'][-1] += mse_left_G.item()
                self.mse_right['AE_G']['train'][-1] += mse_right_G.item()
                self.loss['AE_G']['train'][-1] += loss_G.item()
                loss_G.backward(retain_graph=True)

                mse_left_D = mse(w_D, w).cuda() if self.device == torch.device('cuda') else mse(w_D, w)
                mse_right_D = mse(w_G_D, w).cuda() if self.device == torch.device('cuda') else mse(w_G_D, w)
                loss_D = (1 / epoch) * mse_left_D - (1 - 1 / epoch) * mse_right_D

                # record loss
                self.mse_left['AE_D']['train'][-1] += mse_left_D.item()
                self.mse_right['AE_D']['train'][-1] += mse_right_D.item()
                self.loss['AE_D']['train'][-1] += loss_D.item()
                loss_D.backward()

                self._optimizer_G.step()
                self._optimizer_D.step()

                # valid and log
                if self._step != 0 and self._step % self._valid_step_freq == 0:
                    start_valid = time.time()
                    for w in valid_sliding_window:
                        w = w.view(-1, self._input_dims)
                        w = w.cuda() if self.device == torch.device('cuda') else w

                        z = self._shared_encoder(w)
                        w_G = self._decoder_G(z).cuda() if self.device == torch.device('cuda') else self._decoder_G(z)
                        w_D = self._decoder_D(z).cuda() if self.device == torch.device('cuda') else self._decoder_D(z)
                        w_G_D = self._decoder_D(self._shared_encoder(w_G)).cuda() if self.device == torch.device('cuda') else self._decoder_D(self._shared_encoder(w_G))

                        mse_left_G = mse(w_G, w) if self.device == torch.device('cuda') else mse(w_G, w)
                        mse_right_G = mse(w_G_D, w) if self.device == torch.device('cuda') else mse(w_G_D, w)
                        loss_G = (1 / epoch) * mse_left_G + (1 - 1 / epoch) * mse_right_G
                        
                        mse_left_D = mse(w_D, w) if self.device == torch.device('cuda') else mse(w_D, w)
                        mse_right_D = mse(w_G_D, w) if self.device == torch.device('cuda') else mse(w_G_D, w)
                        loss_D = (1 / epoch) * mse_left_D - (1 - 1 / epoch) * mse_right_D

                        self.mse_left['AE_G']['valid'][-1] += mse_left_G.item()
                        self.mse_right['AE_G']['valid'][-1] += mse_right_G.item()
                        self.loss['AE_G']['valid'][-1] += loss_G.item()
                        self.mse_left['AE_D']['valid'][-1] += mse_left_D.item()
                        self.mse_right['AE_D']['valid'][-1] += mse_right_D.item()
                        self.loss['AE_D']['valid'][-1] += loss_D.item()
#                    print("[Epoch %d/%d][step %d]" % (epoch, self._max_epochs, self._step))
 #                   print(
  #                      "[Train] [AE_G left mse: %f right mse: %f loss: %f][[AE_D left mse: %f right mse: %f loss: %f]"
   #                     % (self.mse_left['AE_G']['train'][-1] / self._valid_step_freq, self.mse_right['AE_G']['train'][-1] / self._valid_step_freq,
    #                       self.loss['AE_G']['train'][-1] / self._valid_step_freq, self.mse_left['AE_D']['train'][-1] / self._valid_step_freq,
     #                      self.mse_right['AE_D']['train'][-1] / self._valid_step_freq, self.loss['AE_D']['train'][-1] / self._valid_step_freq,)
      #              )

       #             print(
        #                "[Valid] [AE_G left mse: %f right mse: %f loss: %f][[AE_D left mse: %f right mse: %f loss: %f]"
         #               % (self.mse_left['AE_G']['valid'][-1] / self._valid_step_freq, self.mse_right['AE_G']['valid'][-1] / self._valid_step_freq,
          #                 self.loss['AE_G']['valid'][-1] / self._valid_step_freq, self.mse_left['AE_D']['valid'][-1] / self._valid_step_freq,
           #                self.mse_right['AE_D']['valid'][-1] / self._valid_step_freq, self.loss['AE_D']['valid'][-1] / self._valid_step_freq,)
            #        )

                    self.mse_left['AE_G']['train'].append(0)
                    self.mse_right['AE_G']['train'].append(0)
                    self.loss['AE_G']['train'].append(0)
                    self.mse_left['AE_D']['train'].append(0)
                    self.mse_right['AE_D']['train'].append(0)
                    self.loss['AE_D']['train'].append(0)

                    self.mse_left['AE_G']['valid'].append(0)
                    self.mse_right['AE_G']['valid'].append(0)
                    self.loss['AE_G']['valid'].append(0)
                    self.mse_left['AE_D']['valid'].append(0)
                    self.mse_right['AE_D']['valid'].append(0)
                    self.loss['AE_D']['valid'].append(0)

                self._step += 1
            
            et_epoch = time.time()
            total_time += (et_epoch - st_epoch)
            # print('-------------------------------------------')
            # print(f'第{epoch}轮训练耗时：{et_epoch - st_epoch}s  总耗时：{total_time}s')
            # print('-------------------------------------------\n')

        print(f'平均每轮训练耗时: {total_time / self._max_epochs}s, 总耗时：{total_time}s')

    def predict(self, values, alpha=0.5, beta=0.5, on_dim=False):
        collect_scores = []
        test_sliding_window = SlidingWindowDataLoader(
            SlidingWindowDataset(values, self._window_size),
            batch_size=self._batch_size,
        )
        mse = nn.MSELoss(reduction='none')
        for w in test_sliding_window:
            w = w.view(-1, self._input_dims)
            w = w.cuda() if self.device == torch.device('cuda') else w

            z = self._shared_encoder(w)
            w_G = self._decoder_G(z).cuda() if self.device == torch.device('cuda') else self._decoder_G(z)
            w_G_D = self._decoder_D(self._shared_encoder(w_G)).cuda() if self.device == torch.device('cuda') else self._decoder_D(self._shared_encoder(w_G))
            batch_scores = alpha * mse(w_G, w).cuda() + beta * mse(w_G_D, w).cuda() if self.device == torch.device('cuda') else alpha * mse(w_G, w) + beta * mse(w_G_D, w)
            batch_scores = batch_scores.view(-1, self._window_size, self._x_dims)
            batch_scores = batch_scores.cuda().data.cpu().numpy() if self.device == torch.device('cuda') else batch_scores.data.numpy()
            if not on_dim:
                batch_scores = np.sum(batch_scores, axis=2)
            if not collect_scores:
                collect_scores.extend(batch_scores[0])
                collect_scores.extend(batch_scores[1:, -1])
            else:
                collect_scores.extend(batch_scores[:, -1])

        return collect_scores

    def reconstruct(self, values):
        collector_1 = []
        collector_2 = []
        test_sliding_window = SlidingWindowDataLoader(
            SlidingWindowDataset(values, self._window_size),
            batch_size=self._batch_size,
        )

        for w in test_sliding_window:
            w = w.view(-1, self._input_dims)
            w = w.cuda() if self.device == torch.device('cuda') else w

            z = self._shared_encoder(w)
            w_G = self._decoder_G(z).cuda() if self.device == torch.device('cuda') else self._decoder_G(z)
            w_G_D = self._decoder_D(self._shared_encoder(w_G)).cuda() if self.device == torch.device('cuda') else self._decoder_D(self._shared_encoder(w_G))
            
            w_G = w_G.view(-1, self._window_size, self._x_dims)
            w_G = w_G.cuda().data.cpu().numpy() if self.device == torch.device('cuda') else w_G.detach().numpy()

            w_G_D = w_G_D.view(-1, self._window_size, self._x_dims)
            w_G_D = w_G_D.cuda().data.cpu().numpy() if self.device == torch.device('cuda') else w_G_D.detach().numpy()

            if not collector_1:
                collector_1.extend(w_G[0])
                collector_1.extend(w_G[1:, -1])
            else:
                collector_1.extend(w_G[:, -1])

            if not collector_2:
                collector_2.extend(w_G_D[0])
                collector_2.extend(w_G_D[1:, -1])
            else:
                collector_2.extend(w_G_D[:, -1])

        return np.array(collector_1), np.array(collector_2)

    def predict_mean(self, value):


        '''
            利用均值作为正常基准
        '''
        def get_percentile_mean(value):
            data = np.asarray(value, dtype=np.float32)
            if len(data.shape) < 2:
                raise ValueError('Data must be a 2-D array')

            t_data = copy.deepcopy(data)
            t_data.sort(axis=1)
            mean = np.mean(t_data[:, int(t_data.shape[1] * 0):int(t_data.shape[1] * 0.8)], axis=1)

            return mean

        scores = np.array(self.predict(value, on_dim=True))
        score_sliding_window = SlidingWindowDataset(scores, 100)

        new_scores = []
        for w in score_sliding_window:
            scaler = get_percentile_mean(w.T)
            new_score = w.T[:, -1] / scaler
            new_scores.append(new_score)
        return np.array(new_scores)

    def predict_distribution(self, value):
        scores = []
        _, AE_2 = self.reconstruct(value)

        for i in range(AE_2.shape[0]):
            temp = []
            for j in range(AE_2.shape[1]):

                # dis = Normal(AE_2[i][j], math.sqrt(AE_2[i][j]))
                temp.append((value[i][j] - AE_2[i][j]) / math.sqrt(AE_2[i][j]))
                # temp.append(dis.log_prob(value[i][j]).exp())
            scores.append(temp)
        return np.array(scores)

    def predict_z(self,value):
        '''
            利用均值作为正常基准
        '''

        def get_mean(value):
            data = np.asarray(value, dtype=np.float32)
            if len(data.shape) < 2:
                raise ValueError('Data must be a 2-D array')

            mean = np.mean(data, axis=1)
            return mean

        def get_var(value):
            data = np.asarray(value, dtype=np.float32)
            if len(data.shape) < 2:
                raise ValueError('Data must be a 2-D array')

            var = np.var(data, axis=1)
            return var

        scores = np.array(self.predict(value, on_dim=True))
        score_sliding_window = SlidingWindowDataset(scores, 100)

        new_scores = []
        m = []
        v = []
        for w in score_sliding_window:
            mean = get_mean(w.T)
            var = get_var(w.T)

            new_score = (w.T[:, -1] - mean) / np.sqrt(var)
            new_scores.append(new_score)
            m.append(var)
            v.append(var)
        return np.array(new_scores), np.array(m), np.array(v)

    def localization(self, scores):
        sort_list = []
        for score in scores:
            contribution = sorted(range(len(score)), key = lambda k: score[k])
            sort_list.append([(i, score[i]) for i in contribution])
        return sort_list

    def save(self, shared_encoder_path, decoder_G_path, decoder_D_path):
        torch.save(self._shared_encoder.state_dict(), shared_encoder_path)
        torch.save(self._decoder_G.state_dict(), decoder_G_path)
        torch.save(self._decoder_D.state_dict(), decoder_D_path)

    def restore(self, shared_encoder_path, decoder_G_path, decoder_D_path):
        self._shared_encoder.load_state_dict(torch.load(shared_encoder_path))
        self._decoder_G.load_state_dict(torch.load(decoder_G_path))
        self._decoder_D.load_state_dict(torch.load(decoder_D_path))
