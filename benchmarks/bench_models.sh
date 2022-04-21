#!/bin/sh


TOTAL_MODELS=40
START=0
END=$(($TOTAL_MODELS-1))

MODEL_TYPE="roberta_63_1.48G"
# MODEL_TYPE="gpt2_2.52M_6.43G"
# MODEL_TYPE="bert_63_1.46G"

MODEL_FILE="sampled_models_$MODEL_TYPE.json"
OUTPUT_FILE="bench_output_$MODEL_TYPE.csv"

MODEL_NAME="chkla/roberta-argument"
python bench_models.py --model_name $MODEL_NAME --model_file $MODEL_FILE
pkill -9 python
python bench_models.py --model_name $MODEL_NAME --model_file $MODEL_FILE --disable_deepspeed
pkill -9 python

exit 0

for i in $(seq $START $END)
do
    python bench_models.py --model_index $i --model_file $MODEL_FILE --output_file $OUTPUT_FILE --reuse_output
    pkill -9 python
    python bench_models.py --model_index $i --model_file $MODEL_FILE --output_file $OUTPUT_FILE --disable_deepspeed --reuse_output
    pkill -9 python
done

