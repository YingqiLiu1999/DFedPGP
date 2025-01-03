#!/bin/bash
python fedml_experiments/standalone/dfedpgp/main_dfedpgp.py \
--model  'resnet18' \
--dataset 'cifar10' \
--partition_method 'n_cls' \
--partition_alpha 2 \
--batch_size 128 \
--client_num_in_total 100 
--frac 0.1 \
--comm_round 500 \
--body_epochs 5 \
--head_epochs 1 \
--lr_body 0.1 \
--lr_head 0.001 \
--lr_decay 0.99 \
--cs 'random' \
--momentum 0.9 \
--seed 0 \
--gpu 6 