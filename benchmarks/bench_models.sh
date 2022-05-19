#!/bin/sh
``
MODEL_TYPE="roberta_81.53M_357.02M_34"
# MODEL_TYPE="gpt2_629.14K_1.61G_12"
# MODEL_TYPE="bert_61.44K_335.54M_40"
# MODEL_TYPE="gpt_neo_83.36M_2.67G_4"
# MODEL_TYPE="gptj_6.04G_6.04G_1"
TOTAL_MODELS=34
START=0
END=$(($TOTAL_MODELS-1))

DATA_TYPE="fp32"
MODEL_FILE="sampled_models_$MODEL_TYPE.json"
OUTPUT_FILE="output_${MODEL_TYPE}_${DATA_TYPE}.csv"

# MODEL_NAME="klue/bert-base"
# python bench_models.py --model_name $MODEL_NAME --model_file $MODEL_FILE
# pkill -9 python
# python bench_models.py --model_name $MODEL_NAME --model_file $MODEL_FILE --disable_deepspeed --reuse_output
# pkill -9 python

# exit 0

for i in $(seq $START $END)
do
    echo "benchmarking model $i"
    python bench_models.py --model_index $i --model_file $MODEL_FILE --output_file $OUTPUT_FILE --reuse_output
    pkill -9 python
    sleep 2
    python bench_models.py --model_index $i --model_file $MODEL_FILE --output_file $OUTPUT_FILE --disable_deepspeed --reuse_output
    pkill -9 python
done
