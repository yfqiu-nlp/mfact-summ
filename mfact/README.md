# Loss-weighting Method for Mitigating Hallucinations

## Installation

Install the environment with the following requirements,

```
python3 -m pip install requirements.txt
```

## Usage

We provide an [example](./xlsum-zh-hallu-classification-input.csv) of inputs for using mFACT as evaluation metrics.

```
lang=zh_CN
python3 transformers/examples/pytorch/text-classification/run_hallu_classification.py \
      --model_name_or_path "mFACT-zh_CN" \
      --do_predict \
      --test_file ${task_dir}"/xlsum-"${lang}"-train-nli-input.csv" \
      --language ${lang:0:2} \
      --train_language ${lang:0:2} \
      --output_dir "dataset/xlsum-weight/"$hallu_classifier"-"$lang"-classifier-train-predictions" \
      --save_steps -1 \
      --overwrite_output_dir \
      --get_predict_scores \
```