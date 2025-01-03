import copy
import logging
import math
import time
import pdb
import numpy as np
import torch

class Client:

    def __init__(self, client_idx, local_training_data, local_test_data, local_sample_number, args, device,
                 model_trainer, logger):
        self.logger = logger
        self.client_idx = client_idx
        self.local_training_data = local_training_data
        self.local_test_data = local_test_data
        self.local_sample_number = local_sample_number
        self.logger.info("self.local_sample_number = " + str(self.local_sample_number))
        self.args = args
        self.device = device
        self.model_trainer = model_trainer

    def update_local_dataset(self, client_idx, local_training_data, local_test_data, local_sample_number):
        self.client_idx = client_idx
        self.local_training_data = local_training_data
        self.local_test_data = local_test_data
        self.local_sample_number = local_sample_number

    def get_sample_number(self):
        return self.local_sample_number

    def train(self, w_global,w_local,round):
        if self.args.dataset == "emnist":
            body_params = {name: copy.deepcopy(w_global[name]) for name in w_global if 'output_layer' not in name}
            head_params = {name: copy.deepcopy(w_local[name]) for name in w_local if 'output_layer' in name}
        else:
            body_params = {name: copy.deepcopy(w_global[name]) for name in w_global if 'linear' not in name}
            head_params = {name: copy.deepcopy(w_local[name]) for name in w_local if 'linear' in name}
        self.model_trainer.set_model_params(body_params)
        self.model_trainer.set_model_params(head_params)
        self.model_trainer.set_id(self.client_idx)
        
        test_local_metrics = self.model_trainer.train \
            (self.local_training_data, self.local_test_data,self.device, self.args, round)
        weights = self.model_trainer.get_model_params()
        return  weights,test_local_metrics

    def local_test(self, w, b_use_test_dataset = True):
        if b_use_test_dataset:
            test_data = self.local_test_data
        else:
            test_data = self.local_training_data
        self.model_trainer.set_model_params(w)
        metrics = self.model_trainer.test(test_data, self.device, self.args)
        return metrics
    
