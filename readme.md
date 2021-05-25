# **Semantic Enhanced Dual-channel Graph Communication Network for Aspect-based Sentiment Analysis**

This work is based on ‘Aspect-Level Sentiment Analysis Via Convolution over Dependency Tree’

### Requirements

pytorch==1.7.0

cuda==11.2

python==3.8

transformers==3.5.1



### Preparation

#### For GloVe

First, download and unzip GloVe vectors(`glove.840B.300d.zip`) from https://nlp.stanford.edu/projects/glove/ .

Then, put `glove.840B.300d.txt` into `./dataset/glove` directory. 

#### For Bert

download and unzip bert model(`bert_base_uncased`) from https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased.tar.gz .

Then, put `config.json` and `pytorch_model.bin` into `./bert_model` directory



### Train

To train the SDGCN model, run the script  in `./script/train` directory



### Eval

To eval the SDGCN model, run the script  in `./script/eval` directory