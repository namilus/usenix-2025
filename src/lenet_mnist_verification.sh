#!/bin/bash

MODEL=lenet
DATASET=mnist
EPOCHS=2
BATCH_SIZE=100

echo "Generating trace for $MODEL/$DATASET..."
TRACE_DIR=$(python3 generate_trace.py $MODEL $DATASET --epochs $EPOCHS --batch-size $BATCH_SIZE --every 1 | tee /dev/tty | tail  -n 1)

echo "Verifying trace saved at $TRACE_DIR..."
VERIF_DIR=$(python3 verify_trace.py $TRACE_DIR --total-runs 1 | tee /dev/tty | tail -n 1)

python3 plot_repr.py $VERIF_DIR
