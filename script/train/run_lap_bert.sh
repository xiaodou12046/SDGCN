#!/bin/bash
python train.py \
--seed 490 \
--threshold 0.1 \
--gcn_dropout 0.3 \
--input_dropout 0.1 \
--num_layers 3 \
--head_num 3 \
--bert_lr 2e-5 \
--hidden_dim 204 \
--rnn_hidden 204 \
--dataset Laptops \
--batch_size 8 \
--lr 0.001 \
--emb_type bert