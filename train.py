#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/9/16 11:22
# @Author  : Huatao
# @Email   : 735820057@qq.com
# @File    : train.py
# @Description :
import copy
import os
import datetime

import numpy as np
import os   
import time
import torch
import torch.nn as nn

from utils import count_model_parameters



class Trainer(object):
    """Training Helper Class"""
    def __init__(self, cfg, model, optimizer, save_path, device):
        self.cfg = cfg  # config for training : see class Config
        self.model = model
        self.optimizer = optimizer
        self.save_path = save_path
        self.device = device  # device name
        
        # Create save directory if it doesn't exist
        save_dir = os.path.dirname(self.save_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
            
        # Create log file path
        run_id = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_filename = f'loss_acc_{run_id}.txt'
        
        # Ensure directory exists for log file
        log_dir = os.path.dirname(self.save_path)
        self.log_path = os.path.join(log_dir, log_filename)
        
        # Create log directory if it doesn't exist
        if not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
            
        print(f"Log file will be saved to: {self.log_path}")

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
              , model_file=None, data_parallel=False, load_self=False): # to be used with triplet loss func

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
        print('Best Accuracy: %0.3f/%0.3f/%0.3f, F1: %0.3f/%0.3f/%0.3f' % best_stat)
    
    def train(self, func_loss, func_forward, func_evaluate, data_loader_train, data_loader_test, data_loader_vali):  # use with semantic loss func
        global_step = 0
        vali_acc_best = 0.0
        best_stat = None
        model_best = self.model.state_dict()

        for e in range(self.cfg.n_epochs):
            # Initialize loss tracking dictionaries
            epoch_losses = {
                'classification_loss': 0.0,
                'semantic_loss': 0.0,
                'triplet_loss': 0.0,
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
                self.save(0)

        self.model.load_state_dict(model_best)
        print('\nTraining completed.')
        print(f'Best Accuracy: Train={best_stat[0]:.3f}, Val={best_stat[1]:.3f}, Test={best_stat[2]:.3f}')
        print(f'Best F1 Score: Train={best_stat[3]:.3f}, Val={best_stat[4]:.3f}, Test={best_stat[5]:.3f}')"""
    
    def train(self, func_loss, func_forward, func_evaluate, data_loader_train, data_loader_test, data_loader_vali): # use with contrastive loss func
        global_step = 0
        vali_acc_best = 0.0
        combined_sore = 0.0
        combined_sore_best = 0.0
        best_stat = None
        model_best = self.model.state_dict()

        for e in range(self.cfg.n_epochs):
            # Track all loss components separately
            epoch_losses = {
                'classification_loss': 0.0,
                'semantic_loss': 0.0,
                'contrastive_loss': 0.0,  # New loss component
                'total_loss': 0.0
            }
            
            self.model.train()
            for i, batch in enumerate(data_loader_train):
                batch = [t.to(self.device) for t in batch]
                self.optimizer.zero_grad()
                
                # Get loss and all its components
                total_loss, batch_losses = func_loss(self.model, batch, current_epoch=e)
                
                # Update running averages for all loss components
                for key in epoch_losses:
                    epoch_losses[key] += batch_losses[key]
                
                total_loss.backward()
                self.optimizer.step()
                global_step += 1

            # Calculate and log average losses
            num_batches = len(data_loader_train)
            avg_losses = {k: v/num_batches for k, v in epoch_losses.items()}
            
            # Evaluate model performance
            train_acc, train_f1 = self.run(func_forward, func_evaluate, data_loader_train)
            test_acc, test_f1 = self.run(func_forward, func_evaluate, data_loader_test)
            vali_acc, vali_f1 = self.run(func_forward, func_evaluate, data_loader_vali)

            # Print detailed epoch results
            print(f'\nEpoch {e+1}/{self.cfg.n_epochs}:')
            print('Loss Components:')
            for loss_name, loss_val in avg_losses.items():
                print(f'  {loss_name}: {loss_val:.4f}')
            print(f'Accuracies: Train={train_acc:.3f}, Val={vali_acc:.3f}, Test={test_acc:.3f}')
            print(f'F1 Scores: Train={train_f1:.3f}, Val={vali_f1:.3f}, Test={test_f1:.3f}')

            # save the loss and accuracy on validation set and the test set as text file as a new file
            """run_id = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            log_filename = f'loss_acc_{run_id}.txt'  # Example: loss_acc_2025-02-11_14-30-00.txt
            self.log_path = os.path.join(self.save_path, log_filename)"""

            # Save log for the run (only one file per run)
            with open(self.log_path, 'a') as f:
                f.write(f'Epoch {e+1}/{self.cfg.n_epochs}:\n')
                f.write('Loss Components:\n')
                for loss_name, loss_val in avg_losses.items():
                    f.write(f'  {loss_name}: {loss_val:.4f}\n')
                f.write(f'Accuracies: Train={train_acc:.3f}, Val={vali_acc:.3f}, Test={test_acc:.3f}\n')
                f.write(f'F1 Scores: Train={train_f1:.3f}, Val={vali_f1:.3f}, Test={test_f1:.3f}\n\n')


            # Save best model based on validation accuracy
            # round the  val accuracy to 2 decimal places
            vali_acc = round(vali_acc, 2)

            combined_sore = (0.6 * vali_acc) + (0.3 * vali_f1) + (0.1 * min(train_f1, 0.99))

            if combined_sore >= combined_sore_best:
                combined_sore_best = combined_sore
                best_stat = (train_acc, vali_acc, test_acc, train_f1, vali_f1, test_f1)
                model_best = copy.deepcopy(self.model.state_dict())
                self.save(0)

        self.model.load_state_dict(model_best)
        print('\nTraining completed.')
        print(f'Best Performance:')
        print(f'Accuracy: Train={best_stat[0]:.3f}, Val={best_stat[1]:.3f}, Test={best_stat[2]:.3f}')
        print(f'F1 Score: Train={best_stat[3]:.3f}, Val={best_stat[4]:.3f}, Test={best_stat[5]:.3f}')

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

