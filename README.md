<div align="center">

<h1>Retrieval-based-Voice-Conversion-WebUI</h1>
An easy-to-use Voice Conversion framework based on VITS.<br><br>

[![madewithlove](https://img.shields.io/badge/made_with-%E2%9D%A4-red?style=for-the-badge&labelColor=orange
)](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI)

<img src="https://counter.seku.su/cmoe?name=rvc&theme=r34" /><br>
  
[![Open In Colab](https://img.shields.io/badge/Colab-F9AB00?style=for-the-badge&logo=googlecolab&color=525252)](https://colab.research.google.com/github/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/blob/main/Retrieval_based_Voice_Conversion_WebUI.ipynb)
[![Licence](https://img.shields.io/github/license/RVC-Project/Retrieval-based-Voice-Conversion-WebUI?style=for-the-badge)](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/blob/main/LICENSE)
[![Huggingface](https://img.shields.io/badge/🤗%20-Spaces-yellow.svg?style=for-the-badge)](https://huggingface.co/lj1995/VoiceConversionWebUI/tree/main/)

[![Discord](https://img.shields.io/badge/RVC%20Developers-Discord-7289DA?style=for-the-badge&logo=discord&logoColor=white)](https://discord.gg/HcsmBBGyVk)

</div>

------
[**Changelog**](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/blob/main/docs/Changelog_EN.md) | [**FAQ (Frequently Asked Questions)**](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/wiki/FAQ-(Frequently-Asked-Questions)) 

[**English**](../en/README.en.md) | [**中文简体**](../../README.md) | [**日本語**](../jp/README.ja.md) | [**한국어**](../kr/README.ko.md) ([**韓國語**](../kr/README.ko.han.md)) | [**Türkçe**](../tr/README.tr.md)


Check our [Demo Video](https://www.bilibili.com/video/BV1pm4y1z7Gm/) here!

Training/Inference WebUI：go-web.bat

![image](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/assets/129054828/00387c1c-51b1-4010-947d-3f3ecac95b87)

Realtime Voice Conversion GUI：go-realtime-gui.bat

![image](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/assets/129054828/143246a9-8b42-4dd1-a197-430ede4d15d7)

> The dataset for the pre-training model uses nearly 50 hours of high quality audio from the VCTK open source dataset.

> High quality licensed song datasets will be added to the training-set often for your use, without having to worry about copyright infringement.

> Please look forward to the pretrained base model of RVCv3, which has larger parameters, more training data, better results, unchanged inference speed, and requires less training data for training.

## Features:
+ Reduce tone leakage by replacing the source feature to training-set feature using top1 retrieval;
+ Easy + fast training, even on poor graphics cards;
+ Training with a small amounts of data (>=10min low noise speech recommended);
+ Model fusion to change timbres (using ckpt processing tab->ckpt merge);
+ Easy-to-use WebUI;
+ UVR5 model to quickly separate vocals and instruments;
+ High-pitch Voice Extraction Algorithm [InterSpeech2023-RMVPE](#Credits) to prevent a muted sound problem. Provides the best results (significantly) and is faster with lower resource consumption than Crepe_full;
+ AMD/Intel graphics cards acceleration supported;
+ Intel ARC graphics cards acceleration with IPEX supported.

## Preparing the environment
The following commands need to be executed with Python 3.8 or higher.

(Windows/Linux)
First install the main dependencies through pip:
```bash
# Install PyTorch-related core dependencies, skip if installed
# Reference: https://pytorch.org/get-started/locally/
pip install torch torchvision torchaudio

#For Windows + Nvidia Ampere Architecture(RTX30xx), you need to specify the cuda version corresponding to pytorch according to the experience of https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/issues/21
#pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117

#For Linux + AMD Cards, you need to use the following pytorch versions:
#pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.4.2
```

Then can use poetry to install the other dependencies:
```bash
# Install the Poetry dependency management tool, skip if installed
# Reference: https://python-poetry.org/docs/#installation
curl -sSL https://install.python-poetry.org | python3 -

# Install the project dependencies
poetry install
```

You can also use pip to install them:
```bash

for Nvidia graphics cards
  pip install -r requirements.txt

for AMD/Intel graphics cards on Windows (DirectML)：
  pip install -r requirements-dml.txt

for Intel ARC graphics cards on Linux / WSL using Python 3.10: 
  pip install -r requirements-ipex.txt

for AMD graphics cards on Linux (ROCm):
  pip install -r requirements-amd.txt
```

------
Mac users can install dependencies via `run.sh`:
```bash
sh ./run.sh
```

## Preparation of other Pre-models
RVC requires other pre-models to infer and train.

```bash
#Download all needed models from https://huggingface.co/lj1995/VoiceConversionWebUI/tree/main/
python tools/download_models.py
```

Or just download them by yourself from our [Huggingface space](https://huggingface.co/lj1995/VoiceConversionWebUI/tree/main/).

Here's a list of Pre-models and other files that RVC needs:
```bash
./assets/hubert/hubert_base.pt

./assets/pretrained 

./assets/uvr5_weights

Additional downloads are required if you want to test the v2 version of the model.

./assets/pretrained_v2

If you want to test the v2 version model (the v2 version model has changed the input from the 256 dimensional feature of 9-layer Hubert+final_proj to the 768 dimensional feature of 12-layer Hubert, and has added 3 period discriminators), you will need to download additional features

./assets/pretrained_v2

#If you are using Windows, you may also need these two files, skip if FFmpeg and FFprobe are installed
ffmpeg.exe

https://huggingface.co/lj1995/VoiceConversionWebUI/blob/main/ffmpeg.exe

ffprobe.exe

https://huggingface.co/lj1995/VoiceConversionWebUI/blob/main/ffprobe.exe

If you want to use the latest SOTA RMVPE vocal pitch extraction algorithm, you need to download the RMVPE weights and place them in the RVC root directory

https://huggingface.co/lj1995/VoiceConversionWebUI/blob/main/rmvpe.pt

    For AMD/Intel graphics cards users you need download:

    https://huggingface.co/lj1995/VoiceConversionWebUI/blob/main/rmvpe.onnx

```

Intel ARC graphics cards users needs to run `source /opt/intel/oneapi/setvars.sh` command before starting Webui.

Then use this command to start Webui:
```bash
python infer-web.py
```

If you are using Windows or macOS, you can download and extract `RVC-beta.7z` to use RVC directly by using `go-web.bat` on windows or `sh ./run.sh` on macOS to start Webui.

## ROCm Support for AMD graphic cards (Linux only)
To use ROCm on Linux install all required drivers as described [here](https://rocm.docs.amd.com/en/latest/deploy/linux/os-native/install.html).

On Arch use pacman to install the driver:
````
pacman -S rocm-hip-sdk rocm-opencl-sdk
````

You might also need to set these environment variables (e.g. on a RX6700XT):
````
export ROCM_PATH=/opt/rocm
export HSA_OVERRIDE_GFX_VERSION=10.3.0
````
Make sure your user is part of the `render` and `video` group:
````
sudo usermod -aG render $USERNAME
sudo usermod -aG video $USERNAME
````
After that you can run the WebUI:
```bash
python infer-web.py
```

# FAQ


Q1:ffmpeg error/utf8 error.

It is most likely not a FFmpeg issue, but rather an audio path issue;

FFmpeg may encounter an error when reading paths containing special characters like spaces and (), which may cause an FFmpeg error; and when the training set’s audio contains Chinese paths, writing it into filelist.txt may cause a utf8 error.
Q2:Cannot find index file after “One-click Training”.

If it displays “Training is done. The program is closed,” then the model has been trained successfully, and the subsequent errors are fake;

The lack of an ‘added’ index file after One-click training may be due to the training set being too large, causing the addition of the index to get stuck; this has been resolved by using batch processing to add the index, which solves the problem of memory overload when adding the index. As a temporary solution, try clicking the “Train Index” button again.
Q3:Cannot find the model in “Inferencing timbre” after training

Click “Refresh timbre list” and check again; if still not visible, check if there are any errors during training and send screenshots of the console, web UI, and logs/experiment_name/*.log to the developers for further analysis.
Q4:How to share a model/How to use others’ models?

The pth files stored in rvc_root/logs/experiment_name are not meant for sharing or inference, but for storing the experiment checkpoits for reproducibility and further training. The model to be shared should be the 60+MB pth file in the weights folder;

In the future, weights/exp_name.pth and logs/exp_name/added_xxx.index will be merged into a single weights/exp_name.zip file to eliminate the need for manual index input; so share the zip file, not the pth file, unless you want to continue training on a different machine;

Copying/sharing the several hundred MB pth files from the logs folder to the weights folder for forced inference may result in errors such as missing f0, tgt_sr, or other keys. You need to use the ckpt tab at the bottom to manually or automatically (if the information is found in the logs/exp_name), select whether to include pitch infomation and target audio sampling rate options and then extract the smaller model. After extraction, there will be a 60+ MB pth file in the weights folder, and you can refresh the voices to use it.
Q5:Connection Error.

You may have closed the console (black command line window).
Q6:WebUI popup ‘Expecting value: line 1 column 1 (char 0)’.

Please disable system LAN proxy/global proxy and then refresh.
Q7:How to train and infer without the WebUI?

Training script:
You can run training in WebUI first, and the command-line versions of dataset preprocessing and training will be displayed in the message window.

Inference script:
https://huggingface.co/lj1995/VoiceConversionWebUI/blob/main/myinfer.py

e.g.

runtime\python.exe myinfer.py 0 “E:\codes\py39\RVC-beta\todo-songs\1111.wav” “E:\codes\py39\logs\mi-test\added_IVF677_Flat_nprobe_7.index” harvest “test.wav” “weights/mi-test.pth” 0.6 cuda:0 True

f0up_key=sys.argv[1]
input_path=sys.argv[2]
index_path=sys.argv[3]
f0method=sys.argv[4]#harvest or pm
opt_path=sys.argv[5]
model_path=sys.argv[6]
index_rate=float(sys.argv[7])
device=sys.argv[8]
is_half=bool(sys.argv[9])
Q8:Cuda error/Cuda out of memory.

There is a small chance that there is a problem with the CUDA configuration or the device is not supported; more likely, there is not enough memory (out of memory).

For training, reduce the batch size (if reducing to 1 is still not enough, you may need to change the graphics card); for inference, adjust the x_pad, x_query, x_center, and x_max settings in the config.py file as needed. 4G or lower memory cards (e.g. 1060(3G) and various 2G cards) can be abandoned, while 4G memory cards still have a chance.
Q9:How many total_epoch are optimal?

If the training dataset’s audio quality is poor and the noise floor is high, 20-30 epochs are sufficient. Setting it too high won’t improve the audio quality of your low-quality training set.

If the training set audio quality is high, the noise floor is low, and there is sufficient duration, you can increase it. 200 is acceptable (since training is fast, and if you’re able to prepare a high-quality training set, your GPU likely can handle a longer training duration without issue).
Q10:How much training set duration is needed?

A dataset of around 10min to 50min is recommended.

With guaranteed high sound quality and low bottom noise, more can be added if the dataset’s timbre is uniform.

For a high-level training set (lean + distinctive tone), 5min to 10min is fine.

There are some people who have trained successfully with 1min to 2min data, but the success is not reproducible by others and is not very informative.
This requires that the training set has a very distinctive timbre (e.g. a high-frequency airy anime girl sound) and the quality of the audio is high; Data of less than 1min duration has not been successfully attempted so far. This is not recommended.
Q11:What is the index rate for and how to adjust it?

If the tone quality of the pre-trained model and inference source is higher than that of the training set, they can bring up the tone quality of the inference result, but at the cost of a possible tone bias towards the tone of the underlying model/inference source rather than the tone of the training set, which is generally referred to as “tone leakage”.

The index rate is used to reduce/resolve the timbre leakage problem. If the index rate is set to 1, theoretically there is no timbre leakage from the inference source and the timbre quality is more biased towards the training set. If the training set has a lower sound quality than the inference source, then a higher index rate may reduce the sound quality. Turning it down to 0 does not have the effect of using retrieval blending to protect the training set tones.

If the training set has good audio quality and long duration, turn up the total_epoch, when the model itself is less likely to refer to the inferred source and the pretrained underlying model, and there is little “tone leakage”, the index_rate is not important and you can even not create/share the index file.
Q12:How to choose the gpu when inferring?

In the config.py file, select the card number after “device cuda:”.

The mapping between card number and graphics card can be seen in the graphics card information section of the training tab.
Q13:How to use the model saved in the middle of training?

Save via model extraction at the bottom of the ckpt processing tab.
Q14:File/memory error(when training)?

Too many processes and your memory is not enough. You may fix it by:

1、decrease the input in field “Threads of CPU”.

2、pre-cut trainset to shorter audio files.
Q15: How to continue training using more data

step1: put all wav data to path2.

step2: exp_name2+path2 -> process dataset and extract feature.

step3: copy the latest G and D file of exp_name1 (your previous experiment) into exp_name2 folder.

step4: click “train the model”, and it will continue training from the beginning of your previous exp model epoch.
Q16: error about llvmlite.dll

OSError: Could not load shared object file: llvmlite.dll

FileNotFoundError: Could not find module lib\site-packages\llvmlite\binding\llvmlite.dll (or one of its dependencies). Try using the full path with constructor syntax.

The issue will happen in windows, install https://aka.ms/vs/17/release/vc_redist.x64.exe and it will be fixed.
Q17: RuntimeError: The expanded size of the tensor (17280) must match the existing size (0) at non-singleton dimension 1. Target sizes: [1, 17280]. Tensor sizes: [0]

Delete the wav files whose size is significantly smaller than others, and that won’t happen again. Than click "train the model"and “train the index”.
Q18: RuntimeError: The size of tensor a (24) must match the size of tensor b (16) at non-singleton dimension 2

Do not change the sampling rate and then continue training. If it is necessary to change, the exp name should be changed and the model will be trained from scratch. You can also copy the pitch and features (0/1/2/2b folders) extracted last time to accelerate the training process.

## Credits
+ [ContentVec](https://github.com/auspicious3000/contentvec/)
+ [VITS](https://github.com/jaywalnut310/vits)
+ [HIFIGAN](https://github.com/jik876/hifi-gan)
+ [Gradio](https://github.com/gradio-app/gradio)
+ [FFmpeg](https://github.com/FFmpeg/FFmpeg)
+ [Ultimate Vocal Remover](https://github.com/Anjok07/ultimatevocalremovergui)
+ [audio-slicer](https://github.com/openvpi/audio-slicer)
+ [Vocal pitch extraction:RMVPE](https://github.com/Dream-High/RMVPE)
  + The pretrained model is trained and tested by [yxlllc](https://github.com/yxlllc/RMVPE) and [RVC-Boss](https://github.com/RVC-Boss).
  
## Thanks to all contributors for their efforts
<a href="https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/graphs/contributors" target="_blank">
  <img src="https://contrib.rocks/image?repo=RVC-Project/Retrieval-based-Voice-Conversion-WebUI" />
</a>

