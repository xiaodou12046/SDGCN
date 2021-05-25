#!/bin/bash
python eval.py --save_dir ./saved_models/glove/best_model_rest.pt \
--head_num 4 --num_layers 3 --threshold 0.3 --hidden_dim 300 \
--rnn_hidden 300 --emb_type glove --dataset Restaurants --batch_size 32