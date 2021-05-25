#!/bin/bash
python eval.py --save_dir ./saved_models/glove/best_model_lap.pt \
--head_num 2 --num_layers 3 --threshold 0.3 --hidden_dim 204 \
--rnn_hidden 204 --emb_type glove --dataset Laptops --batch_size 8