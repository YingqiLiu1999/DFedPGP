import copy
import logging
import pickle
import random

import numpy as np
import torch
import pdb

import os
import gc

from fedml_api.standalone.dfedpgp.client import Client
from fedml_api.model.cv.resnet import  customized_resnet18, tiny_resnet18


class DFedPGPAPI(object):
    def __init__(self, dataset, device, args, model_trainer, logger):
        self.logger = logger
        self.device = device
        self.args = args
        [train_data_num, test_data_num, train_data_global, test_data_global,
         train_data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num] = dataset
        self.train_global = train_data_global
        self.test_global = test_data_global
        self.val_global = None
        self.train_data_num_in_total = train_data_num
        self.test_data_num_in_total = test_data_num
        self.client_list = []
        self.train_data_local_num_dict = train_data_local_num_dict
        self.train_data_local_dict = train_data_local_dict
        self.test_data_local_dict = test_data_local_dict
        self.model_trainer = model_trainer
        self._setup_clients(train_data_local_num_dict, train_data_local_dict, test_data_local_dict, model_trainer)
        self.init_stat_info()

    def _setup_clients(self, train_data_local_num_dict, train_data_local_dict, test_data_local_dict, model_trainer):
        self.logger.info("############setup_clients (START)#############")
        for client_idx in range(self.args.client_num_in_total):
            c = Client(client_idx, train_data_local_dict[client_idx], test_data_local_dict[client_idx],
                       train_data_local_num_dict[client_idx], self.args, self.device, model_trainer, self.logger)
            self.client_list.append(c)
        self.logger.info("############setup_clients (END)#############")

    def train(self):
        w_global = self.model_trainer.get_model_params() # get_model_params
        w_per_mdls = []
  
        for clnt in range(self.args.client_num_in_total):
            w_per_mdls.append(copy.deepcopy(w_global)) #distribute models to each client

        for round_idx in range(self.args.comm_round): 
            print("################Communication round : {}".format(round_idx))
            self.logger.info("################Communication round : {}".format(round_idx))
            
            ## initialize metrics 
            p_test_metrics_before = {
            'num_samples': [],
            'num_correct': [],
            'losses': []}

            w_per_mdls_lstrd = copy.deepcopy(w_per_mdls) 
            
            for clnt_idx in range(self.args.client_num_in_total): 
                self.logger.info('@@@@@@@@@@@@@@@@ Training Client CM({}): {}'.format(round_idx, clnt_idx))
                nei_indexs = []
                
                #select neighbours
                nei_indexs = self._benefit_choose(round_idx, clnt_idx, self.args.client_num_in_total,\
                                                  self.args.client_num_per_round, self.args.cs)
                if self.args.client_num_in_total != self.args.client_num_per_round:
                    nei_indexs = np.append(nei_indexs, clnt_idx)
                nei_indexs = np.sort(nei_indexs) 
                if self.args.cs != "full":
                    self.logger.info("client_indexes = " + str(nei_indexs))
                else:
                    self.logger.info("Choose all clients aka FULLY CONNECTED!")

                # aggregate models
                w_local_mdl = self._aggregate_func(clnt_idx, self.args.client_num_in_total,
                                                            self.args.client_num_per_round, nei_indexs,
                                                            w_per_mdls_lstrd)
                
                # local training 
                client = self.client_list[clnt_idx]
                w_local_mdl,p_test_before_metric = \
                    client.train(copy.deepcopy(w_local_mdl),w_per_mdls[clnt_idx],round_idx)
                w_per_mdls[clnt_idx] = copy.deepcopy(w_local_mdl)
                
                
                ## metrics for each client      
                p_test_metrics_before['num_samples'].append(copy.deepcopy(p_test_before_metric['test_total']))
                p_test_metrics_before['num_correct'].append(copy.deepcopy(p_test_before_metric['test_correct']))
                p_test_metrics_before['losses'].append(copy.deepcopy(p_test_before_metric['test_loss']))
                
                
            ## metrics in each round
            p_test_acc_before = sum(
            [np.array(p_test_metrics_before['num_correct'][i]) / np.array(p_test_metrics_before['num_samples'][i]) for i in
             range(self.args.client_num_in_total)]) / self.args.client_num_in_total
            p_test_loss = sum([np.array(p_test_metrics_before['losses'][i]) / np.array(p_test_metrics_before['num_samples'][i]) for i in
                            range(self.args.client_num_in_total)]) / self.args.client_num_in_total
            stats = {'Local model person_test_acc_before': p_test_acc_before, 'person_test_loss': p_test_loss}
            self.stat_info["person_test_acc_before"].append(p_test_acc_before)
            self.logger.info(stats)
            
            if (round_idx+1) % 50 ==0:
                self.logger.info('person_test_acc_before50={}'.format(self.stat_info["person_test_acc_before"]))  

            ## del  
            del w_local_mdl,p_test_metrics_before,w_per_mdls_lstrd
            gc.collect()
                        
                        
        self.logger.info('person_test_before_acc499={}'.format(self.stat_info["person_test_acc_before"]))  
 
        # show results
        test_max = max(self.stat_info["person_test_acc_before"])*100
        test_index = np.argmax(self.stat_info["person_test_acc_before"])
        stats = {'max person_test_acc_before': test_max, 'index': test_index}
        self.logger.info(stats)
        print("best person_test_acc_before %.3f" %(test_max))
        print("over")



    def _benefit_choose(self, round_idx, cur_clnt, client_num_in_total, client_num_per_round, cs = False):
        if client_num_in_total == client_num_per_round:
            client_indexes = [client_index for client_index in range(client_num_in_total)]
            return client_indexes
        if cs == "random": 
            num_clients = min(client_num_per_round, client_num_in_total)
            np.random.seed(round_idx + cur_clnt)  # make sure for each comparison, we are selecting the same clients each round
            client_indexes = np.random.choice(range(client_num_in_total), num_clients, replace=False)
            while cur_clnt in client_indexes:
                client_indexes = np.random.choice(range(client_num_in_total), num_clients, replace=False)

        elif cs == "ring": 
            left = (cur_clnt - 1 + client_num_in_total) % client_num_in_total
            right = (cur_clnt + 1) % client_num_in_total
            client_indexes = np.asarray([left, right])
            
        elif cs == "grid":
            left = (cur_clnt - 1 + client_num_in_total) % client_num_in_total
            right = (cur_clnt + 1) % client_num_in_total
            top = (cur_clnt - 9 + client_num_in_total) % client_num_in_total
            down = (cur_clnt + 9 + client_num_in_total) % client_num_in_total
            client_indexes = np.asarray([left, right, top, down])

        elif cs =="exp": # (2^6<100<2^7)
            n1 = (cur_clnt + 1 + client_num_in_total) % client_num_in_total
            n2 = (cur_clnt + 2 + client_num_in_total) % client_num_in_total
            n3 = (cur_clnt + 4 + client_num_in_total) % client_num_in_total
            n4 = (cur_clnt + 8 + client_num_in_total) % client_num_in_total
            n5 = (cur_clnt + 16 + client_num_in_total) % client_num_in_total
            n6 = (cur_clnt + 32 + client_num_in_total) % client_num_in_total
            n7 = (cur_clnt + 64 + client_num_in_total) % client_num_in_total
            client_indexes = np.asarray([n1,n2,n3,n4,n5,n6,n7])

        elif cs == "full":
            client_indexes = np.arange(client_num_in_total)
            client_indexes = np.delete(client_indexes, cur_clnt)

        return client_indexes


    def _aggregate(self, w_locals):
        training_num = 0
        for idx in range(len(w_locals)):
            (sample_num, _) = w_locals[idx]
            training_num += sample_num
        w_global ={}
        (sample_num, averaged_params) = w_locals[0]
        for k in averaged_params.keys():
            for i in range(0, len(w_locals)):
                local_sample_number, local_model_params = w_locals[i]
                w = local_sample_number / training_num
                if i == 0:
                    w_global[k] = local_model_params[k] * w
                else:
                    w_global[k] += local_model_params[k] * w
        return w_global

    def _avg_aggregate(self, per_mdls):
        w_tmp = copy.deepcopy(per_mdls[0])
        w = 1 / len(per_mdls)
        for k in w_tmp.keys():
            w_tmp[k] = w_tmp[k] - w_tmp[k]
            for clnt in range(len(per_mdls)):
                w_tmp[k] += per_mdls[clnt][k] * w

        return w_tmp

    def _aggregate_func(self, cur_clnt, client_num_in_total, client_num_per_round, nei_indexs, w_per_mdls_lstrd):
        # self.logger.info('Doing local aggregation!')
        w_tmp = copy.deepcopy(w_per_mdls_lstrd[cur_clnt])
        w = 1 / len(nei_indexs)
        for k in w_tmp.keys():
            w_tmp[k] = w_tmp[k] - w_tmp[k]
            for clnt in nei_indexs:
                w_tmp[k] += w_per_mdls_lstrd[clnt][k] * w

        return w_tmp


    def _test_on_all_clients(self, w_global, w_per_mdls, round_idx):

        self.logger.info("################global_test_on_all_clients : {}".format(round_idx))

        p_test_metrics = {
            'num_samples': [],
            'num_correct': [],
            'losses': []
        }

        for client_idx in range(self.args.client_num_in_total):
            """
            Note: for datasets like "fed_CIFAR100" and "fed_shakespheare",
            the training client number is larger than the testing client number
            """
            # test 
            client = self.client_list[client_idx]
            p_test_local_metrics = client.local_test(w_per_mdls[client_idx], True)
            p_test_metrics['num_samples'].append(copy.deepcopy(p_test_local_metrics['test_total']))
            p_test_metrics['num_correct'].append(copy.deepcopy(p_test_local_metrics['test_correct']))
            p_test_metrics['losses'].append(copy.deepcopy(p_test_local_metrics['test_loss']))

            """
            Note: CI environment is CPU-based computing. 
            The training speed for RNN training is to slow in this setting, so we only test a client to make sure there is no programming error.
            """
            if self.args.ci == 1:
                break
            
        p_test_acc = sum(
            [np.array(p_test_metrics['num_correct'][i]) / np.array(p_test_metrics['num_samples'][i]) for i in
             range(self.args.client_num_in_total)]) / self.args.client_num_in_total
        p_test_loss = sum([np.array(p_test_metrics['losses'][i]) / np.array(p_test_metrics['num_samples'][i]) for i in
                           range(self.args.client_num_in_total)]) / self.args.client_num_in_total

        stats = {'Local model person_test_acc': p_test_acc, 'person_test_loss': p_test_loss}
        self.stat_info["person_test_acc"].append(p_test_acc)
        self.logger.info(stats)


    def init_stat_info(self):
        self.stat_info = {}
        self.stat_info["person_test_acc_before"] = []

