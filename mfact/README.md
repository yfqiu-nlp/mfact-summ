# mFACT for Detecting Hallucinations in Multilingual Summarisation

## Installation

Install the environment with the following requirements,

```
python3 -m pip install requirements.txt
```

## Usage

We provide an [example](./xlsum-zh-hallu-classification-input.csv) of inputs for using mFACT as evaluation metrics.

```
lang=zh_CN

python3 examples/pytorch/text-classification/run_hallu_classification.py \
      --model_name_or_path "mFACT-zh_CN" \
      --do_predict \
      --test_file "xlsum-zh-hallu-classification-input.csv" \
      --language ${lang:0:2} \
      --train_language ${lang:0:2} \
      --output_dir "wl-summ-xlsum-zh_CN" \
      --overwrite_output_dir \
      --get_predict_scores \
```