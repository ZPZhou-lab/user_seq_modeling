#!/bin/bash
deepspeed   --num_gpus=1 train_model_ds.py \
            --deepspeed