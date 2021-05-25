#!/bin/bash
python eval.py --save_dir ./saved_models/bert/best_model_rest.pt \
--head_num 4 --num_layers 2 --threshold 0.5 --hidden_dim 204 \
--rnn_hidden 204 --emb_type bert --dataset Restaurants --batch_size 32