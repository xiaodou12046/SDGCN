#!/bin/bash
python train.py \
--seed 787565331 \
--threshold 0.6 \
--gcn_dropout 0.4 \
--input_dropout 0.7 
--num_layers 4 \
--head_num 4 \
--hidden_dim 300 \
--rnn_hidden 300 \
--dataset Tweets \
--batch_size 32 \
--lr 1e-5 \
--emb_type glove
