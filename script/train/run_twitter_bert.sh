#!/bin/bash
python train.py \
--seed 478 \
--threshold 0.3 \
--gcn_dropout 0.3 \
--input_dropout 0.1 \
--num_layers 3 \
--head_num 3 \
--hidden_dim 204 \
--rnn_hidden 204 \
--dataset Tweets \
--batch_size 32 \
--lr 1e-5 \
--bert_lr 2e-5 \
--emb_type bert