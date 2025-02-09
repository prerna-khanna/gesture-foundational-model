#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/9/16 11:22
# @Author  : Huatao
# @Email   : 735820057@qq.com
# @File    : train.py
# @Description :
import copy
import os
import time

import numpy as np
import torch
import torch.nn as nn

from utils import count_model_parameters


def compute_similarity_matrix(embeddings):
    """Compute normalized similarity matrix from embeddings"""
    # Normalize embeddings
    embeddings = F.normalize(embeddings, p=2, dim=1)
    # Compute similarity matrix
    sim_matrix = torch.matmul(embeddings, embeddings.t())
    return sim_matrix

def compute_distance_matrix(embeddings):
    """Compute normalized distance matrix from embeddings"""
    # Compute pairwise distances
    dist_matrix = torch.cdist(embeddings, embeddings, p=2)
    # Normalize to [0, 1]
    dist_matrix = dist_matrix / dist_matrix.max()
    return dist_matrix


class Trainer(object):
    """Training Helper Class"""
    def __init__(self, cfg, model, optimizer, save_path, device):
        self.cfg = cfg # config for training : see class Config
        self.model = model
        self.optimizer = optimizer
        self.save_path = save_path
        self.device = device # device name
        print(f"Debug - Save path in trainer: {self.save_path}")

    def get_embeddings(self, data_loader):
        """Extract GRU embeddings for all samples"""
        self.model.eval()
        all_embeddings = []
        all_labels = []
        
        with torch.no_grad():
            for batch in data_loader:
                inputs, labels = [t.to(self.device) for t in batch]
                # Get GRU features directly
                embeddings = self.model.gru0(inputs)[0][:, -1, :]  # Last hidden state
                all_embeddings.append(embeddings)
                all_labels.append(labels)
        return torch.cat(all_embeddings, 0), torch.cat(all_labels, 0)

    def pretrain(self, func_loss, func_forward, func_evaluate
              , data_loader_train, data_loader_test, model_file=None, data_parallel=False):
        """ Train Loop """
        self.load(model_file)
        model = self.model.to(self.device)
        if data_parallel: # use Data Parallelism with Multi-GPU
            model = nn.DataParallel(model)

        global_step = 0 # global iteration steps regardless of epochs
        best_loss = 1e6
        model_best = model.state_dict()

        for e in range(self.cfg.n_epochs):
            loss_sum = 0. # the sum of iteration losses to get average loss in every epoch
            time_sum = 0.0
            self.model.train()
            for i, batch in enumerate(data_loader_train):
                batch = [t.to(self.device) for t in batch]
                start_time = time.time()
                self.optimizer.zero_grad()
                loss = func_loss(model, batch)

                loss = loss.mean()# mean() for Data Parallelism
                loss.backward()
                self.optimizer.step()
                time_sum += time.time() - start_time
                global_step += 1
                loss_sum += loss.item()

                # if global_step % self.cfg.save_steps == 0: # save
                #     self.save(global_step)

                if self.cfg.total_steps and self.cfg.total_steps < global_step:
                    print('The Total Steps have been reached.')
                    return
                # print(i)

            loss_eva = self.run(func_forward, func_evaluate, data_loader_test)
            print('Epoch %d/%d : Average Loss %5.4f. Test Loss %5.4f'
                    % (e + 1, self.cfg.n_epochs, loss_sum / len(data_loader_train), loss_eva))
            # print("Train execution time: %.5f seconds" % (time_sum / len(self.data_loader)))
            if loss_eva < best_loss:
                best_loss = loss_eva
                model_best = copy.deepcopy(model.state_dict())
                self.save(0)
        model.load_state_dict(model_best)
        print('The Total Epoch have been reached.')
        # self.save(global_step)

    def run(self, func_forward, func_evaluate, data_loader, model_file=None, data_parallel=False, load_self=False):
        """ Evaluation Loop """
        self.model.eval() # evaluation mode
        self.load(model_file, load_self=load_self)
        # print(count_model_parameters(self.model))
        model = self.model.to(self.device)
        if data_parallel: # use Data Parallelism with Multi-GPU
            model = nn.DataParallel(model)

        results = [] # prediction results
        labels = []
        time_sum = 0.0
        for batch in data_loader:
            batch = [t.to(self.device) for t in batch]
            with torch.no_grad(): # evaluation without gradient calculation
                start_time = time.time()
                result, label = func_forward(model, batch)
                time_sum += time.time() - start_time
                results.append(result)
                labels.append(label)
        # print("Eval execution time: %.5f seconds" % (time_sum / len(dt)))
        if func_evaluate:
            return func_evaluate(torch.cat(labels, 0), torch.cat(results, 0))
        else:
            return torch.cat(results, 0).cpu().numpy()

    """def train(self, func_loss, func_forward, func_evaluate, data_loader_train, data_loader_test, data_loader_vali
              , model_file=None, data_parallel=False, load_self=False):
        ### Train Loop
        self.load(model_file, load_self)
        model = self.model.to(self.device)
        if data_parallel: # use Data Parallelism with Multi-GPU
            model = nn.DataParallel(model)

        global_step = 0 # global iteration steps regardless of epochs
        vali_acc_best = 0.0
        best_stat = None
        model_best = model.state_dict()
        for e in range(self.cfg.n_epochs):
            loss_sum = 0.0 # the sum of iteration losses to get average loss in every epoch
            time_sum = 0.0
            self.model.train()
            for i, batch in enumerate(data_loader_train):
                batch = [t.to(self.device) for t in batch]

                start_time = time.time()
                self.optimizer.zero_grad()
                loss = func_loss(model, batch)

                loss = loss.mean()# mean() for Data Parallelism
                loss.backward()
                self.optimizer.step()

                global_step += 1
                loss_sum += loss.item()
                time_sum += time.time() - start_time
                if self.cfg.total_steps and self.cfg.total_steps < global_step:
                    print('The Total Steps have been reached.')
                    return
            train_acc, train_f1 = self.run(func_forward, func_evaluate, data_loader_train)
            test_acc, test_f1 = self.run(func_forward, func_evaluate, data_loader_test)
            vali_acc, vali_f1 = self.run(func_forward, func_evaluate, data_loader_vali)
            print('Epoch %d/%d : Average Loss %5.4f, Accuracy: %0.3f/%0.3f/%0.3f, F1: %0.3f/%0.3f/%0.3f'
                  % (e+1, self.cfg.n_epochs, loss_sum / len(data_loader_train), train_acc, vali_acc, test_acc, train_f1, vali_f1, test_f1))
            # print("Train execution time: %.5f seconds" % (time_sum / len(self.data_loader)))
            if vali_acc > vali_acc_best:
                vali_acc_best = vali_acc
                best_stat = (train_acc, vali_acc, test_acc, train_f1, vali_f1, test_f1)
                model_best = copy.deepcopy(model.state_dict())
                self.save(0)
        self.model.load_state_dict(model_best)
        print('The Total Epoch have been reached.')
        print('Best Accuracy: %0.3f/%0.3f/%0.3f, F1: %0.3f/%0.3f/%0.3f' % best_stat)"""
    
    def train(self, func_loss, func_forward, func_evaluate, data_loader_train, data_loader_test, data_loader_vali):
        global_step = 0
        vali_acc_best = 0.0
        best_stat = None
        model_best = self.model.state_dict()

        for e in range(self.cfg.n_epochs):
            # Initialize loss tracking dictionaries
            epoch_losses = {
                'classification_loss': 0.0,
                'semantic_loss': 0.0,
                #'triplet_loss': 0.0,
                'total_loss': 0.0
            }
            
            self.model.train()
            for i, batch in enumerate(data_loader_train):
                batch = [t.to(self.device) for t in batch]
                self.optimizer.zero_grad()
                
                # Get loss and components
                total_loss, batch_losses = func_loss(self.model, batch, e)
                
                # Update running averages
                for key in epoch_losses:
                    epoch_losses[key] += batch_losses[key]
                
                total_loss.backward()
                self.optimizer.step()
                global_step += 1

            # Calculate averages
            num_batches = len(data_loader_train)
            avg_losses = {k: v/num_batches for k, v in epoch_losses.items()}
            
            # Evaluate
            train_acc, train_f1 = self.run(func_forward, func_evaluate, data_loader_train)
            test_acc, test_f1 = self.run(func_forward, func_evaluate, data_loader_test)
            vali_acc, vali_f1 = self.run(func_forward, func_evaluate, data_loader_vali)

            # Print epoch results
            print(f'\nEpoch {e+1}/{self.cfg.n_epochs}:')
            print('Losses:')
            for loss_name, loss_val in avg_losses.items():
                print(f'  {loss_name}: {loss_val:.4f}')
            print(f'Accuracies: Train={train_acc:.3f}, Val={vali_acc:.3f}, Test={test_acc:.3f}')
            print(f'F1 Scores: Train={train_f1:.3f}, Val={vali_f1:.3f}, Test={test_f1:.3f}')

            if vali_acc > vali_acc_best:
                vali_acc_best = vali_acc
                best_stat = (train_acc, vali_acc, test_acc, train_f1, vali_f1, test_f1)
                model_best = copy.deepcopy(self.model.state_dict())
                
                # Get embeddings for all datasets
                train_emb, train_labels = self.get_embeddings(data_loader_train)
                val_emb, val_labels = self.get_embeddings(data_loader_vali)
                test_emb, test_labels = self.get_embeddings(data_loader_test)
                
                # Save embeddings and labels
                torch.save({
                    'train_embeddings': train_emb.cpu(),
                    'train_labels': train_labels.cpu(),
                    'val_embeddings': val_emb.cpu(),
                    'val_labels': val_labels.cpu(),
                    'test_embeddings': test_emb.cpu(),
                    'test_labels': test_labels.cpu()
                }, self.save_path + '_embeddings.pt')
                
                self.save(0)
        self.model.load_state_dict(model_best)
        print('\nTraining completed.')
        print(f'Best Accuracy: Train={best_stat[0]:.3f}, Val={best_stat[1]:.3f}, Test={best_stat[2]:.3f}')
        print(f'Best F1 Score: Train={best_stat[3]:.3f}, Val={best_stat[4]:.3f}, Test={best_stat[5]:.3f}')
    
    

    def load(self, model_file, load_self=False):
        """ load saved model or pretrained transformer (a part of model) """
        if model_file:
            print('Loading the model from', model_file)
            if load_self:
                self.model.load_self(model_file + '.pt', map_location=self.device)
            else:
                self.model.load_state_dict(torch.load(model_file + '.pt', map_location=self.device))

    def save(self, i=0):
        """ save current model """
        if i != 0:
            torch.save(self.model.state_dict(), self.save_path + "_" + str(i) + '.pt')
        else:
            torch.save(self.model.state_dict(),  self.save_path + '.pt')

