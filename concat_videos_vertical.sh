model1='/m2v_intern/raozejing/StreamingCode/Streaming5B/Longlive22/vis/inference_dmd_init_only_3th_stage_with_HOIGen_dataset_origin_prompts_infinity_sink_per_block_1-6_588555sink_temporal_scale_memory_tuning/step_000400'
model2='/m2v_intern/raozejing/StreamingCode/Streaming5B/Longlive22/vis/inference_dmd_init_only_3th_stage_with_HOIGen_dataset_origin_prompts_infinity_sink_per_block_1-6_588555sink_temporal_scale_memory_tuning/step_000361'
model3='/m2v_intern/raozejing/StreamingCode/Streaming5B/Longlive22/vis/inference_dmd_init_only_3th_stage_with_HOIGen_dataset_origin_prompts_infinity_sink_per_block_1-6_588555sink_temporal_scale/step_000321'
# model3='/m2v_intern/raozejing/StreamingCode/Streaming5B/Longlive22/vis/inference_dmd_init_only_3th_stage_with_HOIGen_dataset_origin_prompts_infinity_sink5/step_000321'
tag1='+ memory tuning step400'
tag2='+ memory tuning step360'
tag3='ours'
output_dir='/m2v_intern/raozejing/StreamingCode/Streaming5B/Longlive22/vis/memory-tuning-vs-ours'

# python scripts/concat_videos_vertical.py \
#     --models /m2v_intern/raozejing/StreamingCode/Streaming5B/Longlive22/vis/inference_dmd_init_only_3th_stage_with_HOIGen_dataset_origin_prompts_infinity_sink_per_block_1-6_588555sink_temporal_scale/step_000321 \
#     /m2v_intern/raozejing/StreamingCode/Streaming5B/Longlive22/vis/inference_dmd_init_only_3th_stage_with_HOIGen_dataset_origin_prompts_infinity_sink_per_block_1-6_588555sink/step_000321 \
#     /m2v_intern/raozejing/StreamingCode/Streaming5B/Longlive22/vis/inference_dmd_init_only_3th_stage_with_HOIGen_dataset_origin_prompts_infinity_sink5/step_000321 \
#     --tags "Block-Aware Temporal RoPE Scaling" "Dynamic-Allocated sink8+5" "baseline-sink5" \
#     --width 704 \
#     --output_dir /m2v_intern/raozejing/StreamingCode/Streaming5B/Longlive22/vis/rope_scaling-vs-588555-vs-sink5

python scripts/concat_videos_vertical.py \
    --models "$model1" \
    "$model2" \
    "$model3" \
    --tags "$tag1" "$tag2" "$tag3" \
    --width 704 \
    --output_dir "$output_dir"
