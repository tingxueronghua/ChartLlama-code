## ___***ChartLlama: A Multimodal LLM for Chart Understanding and Generation***___

<!-- ### 🔥🔥🔥 The LongerCrafter for longer high-quality video generation are now released! -->

<div align="center">
<!-- <p style="font-weight: bold">
✅ totally <span style="color: red; font-weight: bold">no</span> tuning &nbsp;&nbsp;&nbsp;&nbsp;
✅ less than <span style="color: red; font-weight: bold">20%</span> extra time &nbsp;&nbsp;&nbsp;&nbsp;
✅ support <span style="color: red; font-weight: bold">512</span> frames &nbsp;&nbsp;&nbsp;&nbsp;
</p> -->

 <a href='https://tingxueronghua.github.io/ChartLlama/'><img src='https://img.shields.io/badge/arXiv-2310.15169-b31b1b.svg'></a> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
 <a href='https://tingxueronghua.github.io/ChartLlama/'><img src='https://img.shields.io/badge/Project-Page-Green'></a> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;

 [![Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset-blue)](https://tingxueronghua.github.io/ChartLlama/) &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
 [![Model](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-blue)](https://tingxueronghua.github.io/ChartLlama/)

[**Yucheng Han***](http://tingxueronghua.github.io), [**Chi Zhang***(Corresponding Author)](https://icoz69.github.io/), [Xin Chen](https://chenxin.tech/), [Xu Yang](https://cse.seu.edu.cn/2021/1126/c23024a392593/page.htm), [Zhibin Wang](https://openreview.net/profile?id=~Billzb_Wang1)
<br>
[Gang Yu](https://www.skicyyu.org/), [Bin Fu](https://openreview.net/profile?id=~BIN_FU2), [Hanwang Zhang](https://personal.ntu.edu.sg/hanwangzhang/)
<br><br>
(* equal contributions)

From Tencent and Nanyang Technological University.

<img src=./static/teaser_visualization_final_v3.png>
<!-- <p>Input: "A chihuahua in astronaut suit floating in space, cinematic lighting, glow effect"; 
<br>
Resolution: 1024 x 576; Frames: 64.</p> -->
<!-- <img src=assets/t2v/hd02.gif>
<p>Input: "Campfire at night in a snowy forest with starry sky in the background"; 
<br>
Resolution: 1024 x 576; Frames: 64.</p> -->
</div>
 
## 🔆 Introduction


🤗🤗🤗 We first create an instruction-tuning dataset based on our proposed data generation pipeline. Then, we train ChartLlama on this dataset and achieve the abilities shown in the figure.

### Examples about the abilities of ChartLlama.

<div align="center">
<img src=./static/qualitative_visualization_04.png>
<p>Redraw the chart according to the given chart, and edit the chart following instructions.</p>
</div>

<div align="center">
<img src=./static/qualitative_visualization_05.png>
<p>Draw a new chart based on given raw data and instructions</p>
</div>

## 📝 Changelog
- __[2023.11.27]__: Create the git repository.
<!-- - __[2023.10.25]__: 🔥🔥 Release the 256x256 model and support multi-prompt generation! -->
<br>


<!-- ## 🧰 Models

|Model|Resolution|Checkpoint|Description
|:---------|:---------|:--------|:--------|
|VideoCrafter (Text2Video)|320x512|[Hugging Face](https://huggingface.co/VideoCrafter)|Support 128 frames on NVIDIA A100 (40GB)
|VideoCrafter (Text2Video)|576x1024|[Hugging Face](https://huggingface.co/VideoCrafter/Text2Video-1024-v1.0/blob/main/model.ckpt)|Support 64 frames on NVIDIA A100 (40GB)
|VideoCrafter (Text2Video)|256x256|[Hugging Face](https://huggingface.co/VideoCrafter)|Support 512 frames on NVIDIA A100 (40GB)

(Reduce the number of frames when you have smaller GPUs, e.g. 256x256 resolutions with 64 frames.) -->

## ⚙️ Setup
Refer to the LLaVA-1.5.
<!-- 
### Install Environment via Anaconda (Recommended)
```bash
conda create -n freenoise python=3.8.5
conda activate freenoise
pip install -r requirements.txt
``` -->


## 💫 Inference 

Waiting to be released.


## 📖 TO-DO LIST
- [ ] Open source the training scripts.
- [ ] Open source the evaluation scripts.
- [ ] Open source the inference script.
- [ ] Open source the model and dataset.
- [x] Create the git repository.




## 😉 Citation
```bib
@misc{yucheng2023chartllama,
      title={ChartLlama: A Multimodal LLM for Chart Understanding and Generation}, 
      author={Yucheng Han, Chi Zhang, Xin Chen, Xu Yang, Zhibin Wang, Gang Yu, Bin Fu, Hanwang Zhang},
      year={2023},
      howpublished={\url{https://github.com/tingxueronghua/ChartLlama-code}}
}
```


## 📢 Disclaimer
We develop this repository for RESEARCH purposes, so it can only be used for personal/research/non-commercial purposes.
****
