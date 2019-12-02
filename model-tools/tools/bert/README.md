# Bert Tools

## Transform tensorflow bert model to caffe model

### Prerequisites

1. tensorflow python environment

2. caffe python environment, you need to compile the model-tools project to generate the caffe_pbs2.py in the [Caffe](./Caffe) directory.

### How to use?

Modify the following variables:

1. tensorflow_model_path: tensorflow bert model path

2. seq_length: input word sequence lenth

3. encoder_layers: encoder layer number

4. attention_nums: multi-head attention number

5. caffe_model_path_prefix: caffe result model save path prefix

## Inference Examples

### Noarh TinyBert

we give an example to use bert tools do inference on mobile device in [tinybert](./tinybert). You need to download the Google research bert [modeling.py](https://github.com/google-research/bert/blob/master/modeling.py) from github, and run the [tinybert-infer.py](./tinybert/tinybert-infer.py).
