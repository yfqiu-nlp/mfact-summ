# Detecting and Mitigating Hallucinations in Cross-Lingual Transfer for Abstractive Summarisation

Code and materials for the [paper]() "Detecting and Mitigating Hallucinations in Cross-Lingual Transfer for Abstractive Summarisation". 

Please see the detailed instructions for using our mFACT metrics in [mfact](mfact/), and for using our loss weighing model in [loss-weighting](loss-weighting).

## News

* **mFACT metrics for six languages are now avaliable at HuggingFace! (20/05/2023)**

## Released Mateirals

Here is a quick navigation to all our released materials.

### Translated Faithfulness Classification Datasets

We upload our curated multiligual faithfulness classification dataset in [huggingface](https://huggingface.co/datasets/yfqiu-nlp/mfact-classification).

### mFACT Metrics
<table>
   <tr>
      <td></td>
      <td>Languages</td>
      <td>HF Checkpoint</td>
   </tr>
   <tr>
      <td>mFACT</td>
      <td>Chinese</td>
      <td>https://huggingface.co/yfqiu-nlp/mFACT-zh_CN</td>
   </tr>
   <tr>
      <td></td>
      <td>Spanish</td>
      <td>https://huggingface.co/yfqiu-nlp/mFACT-es_XX</td>
   </tr>
   <tr>
      <td></td>
      <td>French</td>
      <td>https://huggingface.co/yfqiu-nlp/mFACT-fr_XX</td>
   </tr>
   <tr>
      <td></td>
      <td>Hindi</td>
      <td>https://huggingface.co/yfqiu-nlp/mFACT-hi_IN</td>
   </tr>
   <tr>
      <td></td>
      <td>Vietnamese</td>
      <td>https://huggingface.co/yfqiu-nlp/mFACT-vi_VN</td>
   </tr>
   <tr>
      <td></td>
      <td>Turkish</td>
      <td>
      
      [HF link](https://huggingface.co/yfqiu-nlp/mFACT-tr_TR)
      
      </td>
   </tr>
</table>

### Loss-Weighting Summarizer
Please check both our trained langauge and summarisation adapter in  [huggingface](https://huggingface.co/datasets/yfqiu-nlp/mfact-weighted-loss).

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
      <td></td>
   </tr>
   <tr>
      <td>Spanish</td>
      <td>23.49</td>
      <td>23.37</td>
      <td>33.11</td>
      <td></td>
   </tr>
   <tr>
      <td>French</td>
      <td>27.46</td>
      <td>37.11</td>
      <td>40.88</td>
      <td></td>
   </tr>
   <tr>
      <td>Hindi</td>
      <td>24.97</td>
      <td>34.26</td>
      <td>26.46</td>
      <td></td>
   </tr>
   <tr>
      <td>Vietnamese</td>
      <td>28.04</td>
      <td>39.47</td>
      <td>38.20</td>
      <td>
      
      [google dirve](https://drive.google.com/file/d/1sjNv70DVZJKeq_MfJecIynRPA9ucvayZ/view?usp=share_link)
      
      </td>
   </tr>
   <tr>
      <td>Turkish</td>
      <td>17.38</td>
      <td>37.80</td>
      <td>29.20</td>
      <td></td>
   </tr>
</table>

### MAD-X Summarizer


## Citation
