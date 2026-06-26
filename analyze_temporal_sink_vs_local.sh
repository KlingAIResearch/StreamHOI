#!/bin/bash

cd /m2v_intern/raozejing/StreamingCode/Streaming5B/Longlive22

BUFFER_DIR=/m2v_intern_v3/raozejing/vis/vis_sink_local_output_with_bar_5chunk_new
OUT_DIR=${BUFFER_DIR}

python utils/analyze_temporal_sink_vs_local.py \
    --buffer_dir     ${BUFFER_DIR} \
    --output         ${OUT_DIR}/temporal_sink_vs_local_step0.png \
    --denoising_step 0
