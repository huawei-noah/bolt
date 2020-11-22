#!/usr/local/bin/python
# -*- coding: utf-8 -*-


base_params = {
  "sequence.max_length": 15,
  "sequence.num_units": 128,

  "random_seed": 0,
  "use_horovod": False,
  "num_gpus": 8,
  "batch_size_per_gpu": [[300, 400, 500, 600, 700, 800, 900, 1000, 1200, 1400],
                  [120, 90,  72,  60, 50,  44, 38,  20,  18,  18,  18]],
#   "iter_size": 4,
#                          [16,  10,  8,    6,    6,    4]],
  "max_steps": 5000 * 999999,  # 3200 is estimated steps per epoch
  # "num_epochs": 200,

  "save_summaries_steps": 100,
  "print_loss_steps": 50,
  "print_samples_steps": 5000,
  "eval_steps": 5000,
  "save_checkpoint_steps": 5000,
  "num_checkpoints": 6,
  #"logdir": output_dir,

  #"optimizer": NovoGrad,
  "optimizer_params": {
    "beta1": 0.95,
    "beta2": 0.99,
    "epsilon":  1e-08,
    "weight_decay": 1e-04,
    "grad_averaging": False,
    "exclude_from_weight_decay": ["LayerNorm", "layer_norm", "bias", "bn/gamm", "bn/beta"],
   },
  #"lr_policy": poly_decay,
  "lr_policy_params": {
    "learning_rate": 2.4e-2,
    "min_lr": 1.2e-5,
    "power": 2.,
    "warmup_steps": 5000 * 2,
    "begin_decay_at": 5000 * 2,
    "decay_steps": 5000 * 85,
  },
  "larc_params": {
    "larc_eta": 0.001,
  },
  #"dtype": tf.float32,
#   "dtype": "mixed",
#   "loss_scaling": "Backoff",
  # weight decay
#   "regularizer": tf.contrib.layers.l2_regularizer,
#   "regularizer_params": {
#     'scale': 0.0005
#   },
  #"initializer": tf.contrib.layers.xavier_initializer,

  "summaries": ['learning_rate', 'variables', 'gradients', 'larc_summaries',
                'variable_norm', 'gradient_norm', 'global_gradient_norm'],

  "encoder": "ICRNNEncoder",
  "encoder_params": {
#     "use_conv_mask": True,
    "net_blocks": [
        {"conv_type": "conv2d",
         "conv_layers": [
          {
            "states": [1, 1, 128, 1],
            "kernel_size": [3, 7], "stride": [1, 2],
            "num_channels": 32, "padding": "SAME"
          },
          {
            "states": [1, 1, 64, 32],
            "kernel_size": [3, 5], "stride": [2, 2],
            "num_channels": 32, "padding": "SAME"
          },
         ],
        "num_rnn_layers": 0},
        {"conv_type": "conv1d",
         "conv_layers": [
          {
            "kernel_size": [1], "stride": [1],
            "num_channels": 384, "padding": "SAME",
            "activation_fn": None,
          },
         ],
        "num_rnn_layers": 0,
        "transformer_block_params": {
           "n_layer": 2,
           "d_model": 384,
           "n_head": 6,
           "d_head": 64,
           "d_inner": 384 * 4,
           "dropout_keep_prob": 0.9,
           "att_trunc_len": [5, 7],
           "norm_type": 'bn',
           "use_xl_pos_enc": False,
         },
        },
        {"conv_type": "conv1d",
         "conv_layers": [
          {
            "states": [1, 1, 384],
            #"kernel_size": [3, 7], "stride": [1, 2],
            "kernel_size": [3], "stride": [1],
            "num_channels": 1024, "padding": "SAME"
          },
          {
            "states": [1, 1, 1024],
            "kernel_size": [3], "stride": [2],
            "num_channels": 1280, "padding": "SAME"
          },
          {
            "kernel_size": [1], "stride": [1],
            "num_channels": 512, "padding": "SAME",
            "activation_fn": None,
          },
         ],
        "num_rnn_layers": 0,
        "transformer_block_params": {
           "n_layer": 2,
           "d_model": 512,
           "n_head": 8,
           "d_head": 64,
           "d_inner": 512 * 4,
           "dropout_keep_prob": 0.9,
           "att_trunc_len": [7, 9],
           "norm_type": 'bn',
           "use_xl_pos_enc": False,
         },
        },
        {"conv_type": "conv1d",
         "conv_layers": [
          {
            "states": [1, 1, 512],
            "kernel_size": [3], "stride": [1],
            "num_channels": 1024, "padding": "SAME"
          },
          {
            "states": [1, 1, 1024],
            "kernel_size": [3], "stride": [2],
            "num_channels": 1280, "padding": "SAME"
          },
         {
           "kernel_size": [1], "stride": [1],
           "num_channels": 512, "padding": "SAME",
           "activation_fn": None,
         },
         ],
        "num_rnn_layers": 0,
        "transformer_block_params": {
           "n_layer": 4,
           "d_model": 512,
           "n_head": 8,
           "d_head": 64,
           "d_inner": 512 * 4,
           "dropout_keep_prob": 0.9,
           "att_trunc_len": [9, 15, 23, 31],
           "norm_type": 'bn',
           "output_norm_type": 'ln',
           "use_xl_pos_enc": False,
         },
        },
    ], 
        "rnn_cell_dim": 0,

    "use_cudnn_rnn": False,
    #"rnn_type": "cudnn_gru",
    "rnn_type": "omni_lstm",
    "rnn_unidirectional": True,

    "row_conv": False,
#     "row_conv_width": 3,
    "output_fc": False,

    "n_hidden": 512,

    "dropout_keep_prob": 0.85,
    "activation_fn": "relu",
    # "data_format": "BCFT", # "channels_first",'BCTF', 'BTFC', 'BCFT', 'BFTC'
  },

  "decoder": "TransducerDecoder",
  "decoder_params": {
    "use_sa_pred_net": True,
    "blank_last": True,
    "prepend_start_token": True,
    "start_token_id": 1,
    "pred_net_params": {
#       'emb_drop_word': True,
      'emb_keep_prob': 0.9,
      "emb_size": 128,
      'mask_label_prob': 0.1,
      'mask_label_id': 3,
      'norm_inputs': False,
      "transformer_block_params": {
        "n_layer": 4,
        "d_model": 512,
        "n_head": 8,
        "d_head": 64,
        "d_inner": 512 * 4,
        "dropout_keep_prob": 0.9,
        "input_keep_prob": 1.0,
        "input_project": True,
        "att_trunc_len": [3, 5, 7, 9],
        "norm_type": 'bn',
        "output_norm_type": 'ln',
        "use_xl_pos_enc": False,
      },
    },

    "joint_net_params": {
      "hidden_units": 512,
      "activation_fn": "relu",
      "tie_embedding": False,
    },
  },

  "data_layer": "Speech2TextDataLayer",
  "data_layer_params": {
    #"data_dir": data_dir,
    "bpe": False,
    "text_type": 'zh_sep',
    #"vocab_file": transcript_data_dir + "/pinyin/pinyin_vocab.txt",
    
    "feat_pad_value": -4.0,
    
    #"features_mean_path": transcript_data_dir + "/feat_stats_128_gain_mel/feat_mean.npy",
    #"features_std_dev_path": transcript_data_dir + "/feat_stats_128_gain_mel/feat_std.npy",

    "num_audio_features": 128,
    "input_type": "logfbank",
    "norm_per_feature": True,
    "window": "hanning",
    "precompute_mel_basis": True,
    "sample_freq": 16000,
    "gain": 1.0/32767.0,
    "pad_to": 16,
#     "dither": 1e-5,
    "backend": "librosa"
  },

  #"loss": FakeTransducerLoss,
  "loss_params": {},
}
