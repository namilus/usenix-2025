#!/bin/bash

MODEL=lenet
DATASET=mnist
EPOCHS=2
BATCH_SIZE=100

echo "Generating trace for $MODEL/$DATASET..."
TRACE_DIR=$(python3 generate_trace.py $MODEL $DATASET --epochs $EPOCHS --batch-size $BATCH_SIZE --every 1 | tee /dev/tty | tail  -n 1)


NUM_TO_FORGE=1
THRESHOLD=1e-7 # taken from our results (see Fig 2 for Lenet/mnist b = 100)

FORGE_RESULTS=$(python3 forge_across_checkpoints.py $TRACE_DIR -f $NUM_TO_FORGE --every 100 --threshold $THRESHOLD | tee /dev/tty | tail  -n 1)

python3 plot_forging.py $FORGE_RESULTS
