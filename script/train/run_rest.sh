#!/bin/bash
python train.py \
--seed 413913740 \
--threshold 0.3 \
--gcn_dropout 0.4 \
--input_dropout 0.66 \
--num_layers 3 \
--head_num 4 \
--hidden_dim 300 \
--rnn_hidden 300 \
--dataset Restaurants \
--batch_size 32 \
--lr 0.001 \
--emb_type glove