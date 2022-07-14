# MII Benchmarks

This folder contains assets of selected models for benchmarking and the corresponding run code.

## Description

The selected models are in the `sampled_xxx.json` files. The main logic for benchmarking is in `bench_models.py`

## Getting Started

### Dependencies

See requirements.txt

### How to run

you need to install `Deepspeed-MII` before using this benchmark code.


* Set `MODEL_TYPE` and `TOTAL_MODELS` to what model type and the number of model to evaluate (the total number of models for each type is the last word of MODEL_TYPE), then run

```
./bench_models.sh
```
* After the run finishes, the result is in `OUTPUT_FILE` defined in the `bench_models.sh`.

The text-generation task by default measures `20` token generation with a fixed input string (5-token).