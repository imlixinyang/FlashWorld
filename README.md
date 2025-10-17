
<p align="center">
  <h2 align="center">
        <img src="https://github.com/imlixinyang/FlashWorld-Project-Page/blob/main/static/images/favicon.svg" alt="FlashWorld" style="height: 1.2rem; width: auto; margin-right: -2rem; vertical-align: middle;">
        <em>FlashWorld: High-quality 3D Scene Generation within Seconds</em></h2>

  <p align="center">
        <a href="https://arxiv.org/pdf/2510.13678"><img src='https://img.shields.io/badge/arXiv-FlashWorld-red?logo=arxiv' alt='Paper PDF'></a>
        <a href='https://imlixinyang.github.io/FlashWorld-Project-Page'><img src='https://img.shields.io/badge/Project_Page-FlashWorld-green' alt='Project Page'></a>
        <a href='[https://imlixinyang.github.io/FlashWorld-Project-Page](https://huggingface.co/spaces/imlixinyang/FlashWorld-Demo-Spark)'><img src='https://img.shields.io/badge/Huggingface-Online_Demo-yellow' alt='Online Demo'></a>
        <!-- <a href='https://colab.research.google.com/drive/1LtnxgBU7k4gyymOWuonpOxjatdJ7AI8z?usp=sharing'><img src='https://img.shields.io/badge/Colab_Demo-Director3D-yellow?logo=googlecolab' alt='Project Page'></a> -->
  </p>


  <p align="center">
  <img width="3182" height="1174" alt="teaser" src="https://github.com/user-attachments/assets/e4aae261-83fd-494d-9b08-00ae265a74e4" />
  </p>


***TL;DR:*** FlashWorld enables fast (**7 seconds on a single A100/A800 GPU**) and high-quality 3D scene generation across diverse scenes, from a single image or text prompt.

FlashWorld also supports generation with only **24GB** GPU memory!

## Demo

https://github.com/user-attachments/assets/12ba4776-e7b7-4152-b885-dd6161aa9b4b

## 🔥 News:

The code is actively updated. Please stay tuned!

- [2025.10.17] We release an online demo on Huggingface Spaces at [FlashWorld Online Demo](https://huggingface.co/spaces/imlixinyang/FlashWorld-Demo-Spark).

- [2025.10.16] Paper and local demo code released.

## Installation

- install packages
```
pip install torch torchvision
pip install triton transformers pytorch_lightning omegaconf ninja numpy jaxtyping rich tensorboard einops moviepy==1.0.3 webdataset accelerate opencv-python lpips av plyfile ftfy peft tensorboard pandas flask
```

Please refer to the `requirements.txt` file for the exact package versions.

- install ```gsplat@1.5.2``` and ```diffusers@wan-5Bi2v``` packages
```
pip install git+https://github.com/nerfstudio-project/gsplat.git@32f2a54d21c7ecb135320bb02b136b7407ae5712
pip install git+https://github.com/huggingface/diffusers.git@447e8322f76efea55d4769cd67c372edbf0715b8
```

- clone this repo:
```
git clone https://github.com/imlixinyang/FlashWorld.git
cd FlashWorld
```

- run our demo app by:
```
python app.py
```

If your machine has limited GPU memory, consider adding the ```--offload_t5``` flag to offload text encoding to the CPU, which will reduce GPU memory usage. Note that this may slow down the generation speed somewhat.

Then, open your web browser and navigate to ```http://HOST_IP:7860``` to start exploring FlashWorld!

<!-- We also provide example trajectory josn files and input images in the `examples/` directory. -->
  
## More Generation Results

[https://github.com/user-attachments/assets/bbdbe5de-5e15-4471-b380-4d8191688d82](https://github.com/user-attachments/assets/53d41748-4c35-48c4-9771-f458421c0b38)

## Citation

```
@misc{li2025flashworld,
        title={FlashWorld: High-quality 3D Scene Generation within Seconds},
        author={Xinyang Li and Tengfei Wang and Zixiao Gu and Shengchuan Zhang and Chunchao Guo and Liujuan Cao},
        year={2025},
        eprint={2510.13678},
        archivePrefix={arXiv},
        primaryClass={cs.CV}
    }
```


## License

Licensed under the CC BY-NC-SA 4.0 (Attribution-NonCommercial-ShareAlike 4.0 International)

The code is released for academic research use only. 

If you have any questions, please contact me via [imlixinyang@gmail.com](mailto:imlixinyang@gmail.com). 

