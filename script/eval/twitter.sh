#!/bin/bash
python eval.py --save_dir ./saved_models/glove/best_model_tweet.pt \
--head_num 4 --num_layers 4 --threshold 0.6 --hidden_dim 300 \
--rnn_hidden 300 --emb_type glove --dataset Tweets --batch_size 32