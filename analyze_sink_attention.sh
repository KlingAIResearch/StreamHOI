#!/bin/bash

cd /m2v_intern/raozejing/StreamingCode/Streaming5B/Longlive22

OUT_DIR=/m2v_intern/raozejing/StreamingCode/Streaming5B/Longlive22/vis/vis_sink_output_with_zhexiantu_5chunk

# for STEP in 0 1 2 3; do
#     python utils/analyze_sink_attention_multi.py \
#         --buffer_dir /m2v_intern/raozejing/StreamingCode/Streaming5B/Longlive22/vis/vis_sink_output_with_zhexiantu_5chunk \
#         --mask_dir   /m2v_intern/raozejing/StreamingCode/dataset/live_5s_final_mix_HOIGen_origin_prompt/testset_human-object-masks-first-frame \
#         --output     ${OUT_DIR}/block_sink_attention_human_object_step${STEP}.png \
#         --denoising_step $STEP
# done

python utils/analyze_sink_attention_multi.py \
        --buffer_dir /m2v_intern/raozejing/StreamingCode/Streaming5B/Longlive22/vis/vis_sink_output_with_zhexiantu_5chunk \
        --mask_dir   /m2v_intern/raozejing/StreamingCode/dataset/live_5s_final_mix_HOIGen_origin_prompt/testset_human-object-masks-first-frame \
        --output     ${OUT_DIR}/block_sink_attention_human_object_step${STEP}.png \
        --denoising_step 0