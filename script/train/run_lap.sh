#!/bin/bash
python train.py \
--seed 489 \
--threshold 0.3 \
--gcn_dropout 0.6 \
--input_dropout 0.7 \
--num_layers 3 \
--head_num 2 \
--hidden_dim 204 \
--rnn_hidden 204 \
--dataset Laptops \
--batch_size 8 \
--lr 1e-4 \
--emb_type glove