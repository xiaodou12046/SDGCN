#!/bin/bash
python train.py \
--seed 228 \
--threshold 0.5 \
--gcn_dropout 0.18 \
--input_dropout 0.1 \
--num_layers 2 \
--head_num 4 \
--hidden_dim 204 \
--rnn_hidden 204 \
--dataset Restaurants 
--batch_size 32 \
--lr 1e-5 \
--bert_lr 5e-5 \
--emb_type bert