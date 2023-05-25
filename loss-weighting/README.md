# Loss-weighting Method for Mitigating Hallucinations

## Installation

Install the environment with the following requirements,

```
python3 -m pip install requirements.txt
mkdir dataset
mkdir models
```

We use the code from `adapter-transformers` for our implementaiton of adapters in MAD-X and loss-weighing model. Thanks for their nice work!

## Usage

You should first download our trained model checkpoint from [here](#checkpoints-and-outputs), and place all files to `./models`. You should download this checkpoint for either training or testing, because it contains all language adapters we used.

### Model Inference

To use our model for inference on `TGT` language, please follow the example for Chinese (zh_CN) here,

```
# All language signs: zh_CN, es_XX, fr_XX, hi_IN, vi_VN, tr_TR

TGT="zh_CN"
pretrained="facebook/mbart-large-50"

python3 adapter-transformers/examples/pytorch/summarization/run_summarization.py \
    --model_name_or_path ${pretrained} \
    --do_predict \
    --test_file "dataset/xlsum-"${TGT}"-test.csv" \
    --lang ${TGT} \
    --forced_bos_token ${TGT} \
    --output_dir "models/zs-decoding-xlsum-"${TGT} \
    --overwrite_output_dir \
    --predict_with_generate \
    --per_device_eval_batch_size=32 \
    --generation_max_length 84 \
    --generation_num_beams 6 \
    --length_penalty 0.6 \
    --min_length 30 \
    --no_repeat_ngram_size 3 \
    --train_adapter \
    --load_adapter "models/summarization" \
    --load_lang_adapter "models/"$TGT"-mlm"
```

### Model Training

If you want to use our processed dataet for xlsum English split with weighting score, please download the dataset [here](https://huggingface.co/datasets/yfqiu-nlp/xlsum-en_XX-weights) and place the `xlsum-en_XX-*.csv` file to `./dataset/xlsum-weight/`. 

To train weighting-loss model on `SRC` language, please follow the example for English (en_XX) here,

```
SRC="en_XX"

pretrained="facebook/mbart-large-50"
MODEL_OUTPUT="models/"

python3 adapter-transformers/examples/pytorch/summarization/run_summarization_ours.py \
    --model_name_or_path ${pretrained} \
    --do_train \
    --do_eval \
    --do_predict \
    --train_file dataset/xlsum-weight/xlsum-en_XX-train.csv" \
    --validation_file "dataset/xlsum-weight/xlsum-en_XX-validation.csv" \
    --test_file "dataset/xlsum-weight/xlsum-en_XX-test.csv" \
    --text_column "document" \
    --summary_column "summary" \
    --weight_column "weight" \
    --lang ${SRC} \
    --forced_bos_token ${SRC} \
    --num_train_epochs 10 \
    --output_dir $MODEL_OUTPUT \
    --gradient_accumulation_steps 8 \
    --per_device_train_batch_size=4 \
    --per_device_eval_batch_size=16 \
    --overwrite_output_dir \
    --predict_with_generate \
    --warmup_steps 500 \
    --learning_rate 1e-04 \
    --label_smoothing_factor 0.1 \
    --weight_decay 0.01 \
    --max_grad_norm 0.1 \
    --lr_scheduler_type polynomial \
    --logging_strategy epoch \
    --save_strategy epoch \
    --generation_max_length 60 \
    --generation_num_beams 6 \
    --length_penalty 1.0 \
    --min_length 10 \
    --no_repeat_ngram_size 3 \
    --train_adapter \
    --load_lang_adapter "models/"$SRC"-mlm" \
    --weighted_loss \
```

You are expected to train a model with the similar performance as our following tables. Note that we select the model checkpoint based on the best validation mFACT-English score.

## Checkpoints and Outputs

### Loss-Weighting Summarizer
Please check both our trained language and summarisation adapter in [huggingface](https://huggingface.co/yfqiu-nlp/mfact-weighted-loss).

<table>
   <tr>
      <td>Languages</td>
      <td>R-1</td>
      <td>mFACT</td>
      <td>mFACT-Transfer</td>
      <td>Outputs</td>
   </tr>
   <tr>
      <td>Chinese</td>
      <td>31.25</td>
      <td>42.97</td>
      <td>36.02</td>
      <td>
        <a href="https://drive.google.com/file/d/1oIHJMn7zwxKWrayNxiU42aduQw5QQ4hT/view?usp=share_link">
            Google drive
        </a>
      </td>
   </tr>
   <tr>
      <td>Spanish</td>
      <td>23.49</td>
      <td>23.37</td>
      <td>33.11</td>
      <td>
        <a href="https://drive.google.com/file/d/1esq1joNOu71binoFgHL7JuOrvy4JhKV0/view?usp=share_link">
            Google drive
        </a>
      </td>
   </tr>
   <tr>
      <td>French</td>
      <td>27.46</td>
      <td>37.11</td>
      <td>40.88</td>
      <td>
        <a href="https://drive.google.com/file/d/1khWkb5gKD4tdjmet9IEjEhmRM8Ik-lAF/view?usp=share_link">
            Google drive
        </a>
      </td>
   </tr>
   <tr>
      <td>Hindi</td>
      <td>24.97</td>
      <td>34.26</td>
      <td>26.46</td>
      <td>
        <a href="https://drive.google.com/file/d/1esMLR9bJKhRpcHJgZfadNz98tsWU-0F7/view?usp=share_link">
            Google drive
        </a>
      </td>
   </tr>
   <tr>
      <td>Vietnamese</td>
      <td>28.04</td>
      <td>39.47</td>
      <td>38.20</td>
      <td>
        <a href="https://drive.google.com/file/d/1sjNv70DVZJKeq_MfJecIynRPA9ucvayZ/view?usp=share_link">
            Google drive
        </a>
      </td>
   </tr>
   <tr>
      <td>Turkish</td>
      <td>17.38</td>
      <td>37.80</td>
      <td>29.20</td>
      <td>
        <a href="https://drive.google.com/file/d/1U2X5wqBoHQTipdtj-VDqMRgv9PAvOk28/view?usp=share_link">
            Google drive
        </a>
      </td>
   </tr>
</table>

### MAD-X Summarizer

Please check both our trained language and summarisation adapter in  [huggingface](https://huggingface.co/yfqiu-nlp/mfact-mad-x).

<table>
   <tr>
      <td>Languages</td>
      <td>R-1</td>
      <td>mFACT</td>
      <td>mFACT-Transfer</td>
      <td>Outputs</td>
   </tr>
   <tr>
      <td>Chinese</td>
      <td>28.97</td>
      <td>34.58</td>
      <td>30.62</td>
      <td>
        <a href="https://drive.google.com/file/d/1goEapvFahhovxtYKpDWa_5XdIPjukqwk/view?usp=share_link">
            Google drive
        </a>
      </td>
   </tr>
   <tr>
      <td>Spanish</td>
      <td>22.83</td>
      <td>21.24</td>
      <td>29.28</td>
      <td>
        <a href="https://drive.google.com/file/d/17U1YHp_QOXLWstV_EGe10lpatQoHTipp/view?usp=share_link">
            Google drive
        </a>
      </td>
   </tr>
   <tr>
      <td>French</td>
      <td>25.80</td>
      <td>40.24</td>
      <td>43.64</td>
      <td>
        <a href="https://drive.google.com/file/d/1OIOgQhU-8ajlvG-Cs0tmCoQJo-NSaTwG/view?usp=share_link">
            Google drive
        </a>
      </td>
   </tr>
   <tr>
      <td>Hindi</td>
      <td>24.79</td>
      <td>25.74</td>
      <td>16.89</td>
      <td>
        <a href="https://drive.google.com/file/d/1XXgWsamD5DwKinqFnfF_WOldzVUd2N_R/view?usp=share_link">
            Google drive
        </a>
      </td>
   </tr>
   <tr>
      <td>Vietnamese</td>
      <td>26.97</td>
      <td>34.59</td>
      <td>35.21</td>
      <td>
        <a href="https://drive.google.com/file/d/1cZJLIJrT40domeCK_PoCDmU1CM3qzByu/view?usp=share_link">
            Google drive
        </a>
      </td>
   </tr>
   <tr>
      <td>Turkish</td>
      <td>17.05</td>
      <td>32.15</td>
      <td>22.63</td>
      <td>
        <a href="https://drive.google.com/file/d/1mEya871CsO5PzQQ2MLzW-dOiJQopMQ5j/view?usp=share_link">
            Google drive
        </a>
      </td>
   </tr>
</table>
