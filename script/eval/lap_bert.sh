#!/bin/bash
python eval.py --save_dir ./saved_models/bert/best_model_lap_bert.pt \
--head_num 3 --num_layers 3 --threshold 0.1 --hidden_dim 204 \
--rnn_hidden 204 --emb_type bert --dataset Laptops --batch_size 8