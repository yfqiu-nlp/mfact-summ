# mFACT for Detecting Hallucinations in Multilingual Summarisation

## Installation

Install the environment with the following requirements,

```
python3 -m pip install requirements.txt
```
## Dataset for Training mFACT

You can download our synthesised dataset for training mFACT from [google cloud](https://drive.google.com/drive/folders/1WbTqPWLGTGkJFgVxVehPMjM0rYGUwwG7?usp=sharing) for `en, es, fr, zh, hi, tr, vi`.

## Usage

### Training mFACT

```
train_lang=zh_CN

python examples/pytorch/text-classification/run_hallu_classification.py \
  --model_name_or_path ${pretrained} \
  --do_train \
  --do_eval \
  --do_predict \
  --train_file "dataset/xsum-translation/"$train_lang"-translated-train.csv" \
  --validation_file "dataset/xsum-translation/"$train_lang"-translated-valid.csv" \
  --test_file "dataset/xsum-translation/"$train_lang"-translated-test.csv" \
  --language $train_lang \
  --train_language $train_lang \
  --per_device_train_batch_size 32 \
  --learning_rate 5e-5 \
  --num_train_epochs 2.0 \
  --max_seq_length 512 \
  --output_dir "wl-summ-xlsum-zh_CN" \
  --save_steps -1 \
  --overwrite_output_dir \
```


### Inference

We provide an [example](./xlsum-zh-hallu-classification-input.csv) of inputs for using mFACT in as evaluation metrics.

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