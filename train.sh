#!/bin/bash

# Define variables
x_size=128
y_size=128
hidden_channels=64
hidden_size=128
flow_depth=8
num_levels=3
learn_top=True
dataset_root="Data"
num_classes=2
y_bits=1
learning_rate=0.0002
num_steps=50000
batch_size=4


# Run the Python script with the specified variables
python training.py \
  --x_size "$x_size" \
  --y_size "$y_size" \
  --hidden_channels $hidden_channels \
  --hidden_size $hidden_size \
  -K $flow_depth \
  -L $num_levels \
  --learn_top $learn_top \
  -r "$dataset_root" \
  --num_classes $num_classes \
  --y_bits $y_bits \
  --lr $learning_rate \
  --num_steps $num_steps \
  --batch_size $batch_size \