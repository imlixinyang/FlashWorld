
<p align="center">
  <h2 align="center">
        <img src="https://github.com/imlixinyang/FlashWorld-Project-Page/blob/main/static/images/favicon.svg" alt="FlashWorld" style="height: 1.2rem; width: auto; margin-right: -2rem; vertical-align: middle;">
        <em>FlashWorld: High-quality 3D Scene Generation within Seconds</em></h2>

  <p align="center">
        <a href="https://arxiv.org/pdf/2510.13678"><img src='https://img.shields.io/badge/arXiv-FlashWorld-red?logo=arxiv' alt='Paper PDF'></a>
        <a href='https://imlixinyang.github.io/FlashWorld-Project-Page'><img src='https://img.shields.io/badge/Project_Page-FlashWorld-green' alt='Project Page'></a>
        <a href='https://huggingface.co/spaces/imlixinyang/FlashWorld-Demo-Spark'><img src='https://img.shields.io/badge/Huggingface-Online_Demo-yellow' alt='Online Demo'></a>
        <!-- <a href='https://colab.research.google.com/drive/1LtnxgBU7k4gyymOWuonpOxjatdJ7AI8z?usp=sharing'><img src='https://img.shields.io/badge/Colab_Demo-Director3D-yellow?logo=googlecolab' alt='Project Page'></a> -->
  </p>


  <p align="center">
  <img width="3182" height="1174" alt="teaser" src="https://github.com/user-attachments/assets/e4aae261-83fd-494d-9b08-00ae265a74e4" />
  </p>


***TL;DR:*** FlashWorld enables fast (**7 seconds on a 1x A100/A800 GPU, 4 seconds on 1x H100/H800 GPU**) and high-quality 3D scene generation across diverse scenes, from a single image or text prompt.

## Demo

https://github.com/user-attachments/assets/12ba4776-e7b7-4152-b885-dd6161aa9b4b

## 🔥 News:

The code is actively updated. Please stay tuned!

- [2025.10.19] We release a command line interface (CLI). If you want to achieve the **best** generation results with FlashWorld, you can create a good input JSON using the Web Interface and then use the CLI to regenerate and render the scene.

- [2025.10.19] We update the web interface for better logging and faster downloading.

- [2025.10.17] We release an online demo on Huggingface Spaces at [FlashWorld Online Demo](https://huggingface.co/spaces/imlixinyang/FlashWorld-Demo-Spark).

- [2025.10.16] Paper and local demo code released.

## Installation

- install packages
```
pip install torch torchvision
pip install triton transformers omegaconf ninja numpy jaxtyping rich einops moviepy==1.0.3 accelerate opencv-python av plyfile ftfy pandas uvicorn nanobind
```

Please refer to the `requirements.txt` file for the exact package versions.

- install ```gsplat@1.5.2```, ```diffusers@wan-5Bi2v``` and ```spz``` packages
```
pip install git+https://github.com/nerfstudio-project/gsplat.git@32f2a54d21c7ecb135320bb02b136b7407ae5712
pip install git+https://github.com/huggingface/diffusers.git@447e8322f76efea55d4769cd67c372edbf0715b8
pip install git+https://github.com/nianticlabs/spz.git@a4fc69e7948c7152e807e6501d73ddc9c149ce37
```

- (optional) install [sage-attention](https://github.com/thu-ml/SageAttention) package.
```
git clone https://github.com/thu-ml/SageAttention.git
cd SageAttention 
export EXT_PARALLEL=4 NVCC_APPEND_FLAGS="--threads 8" MAX_JOBS=32 # parallel compiling (Optional)
python setup.py install  # or pip install -e .
```

- clone this repo:
```
git clone https://github.com/imlixinyang/FlashWorld.git
cd FlashWorld
```

## Local Web Interface

```
python app.py
```
Then, open your web browser and navigate to ```YOUR_ADDRESS:7860/app``` to start exploring FlashWorld!

If your machine does not have enough GPU memory, add the ```--offload_t5``` and ```--offload_transformer_during_vae``` flags to offload text encoding to the CPU, which will reduce GPU memory usage with little impact on generation speed.
You can also add the ```--offload_vae``` flag, which will greatly reduce GPU memory usage to below 10GB, but will significantly increase generation time. Please use this flag with caution.


On a single A800 GPU, the generation time and GPU memory usage under different settings are as follows:

| Generation Time       | GPU Memory | Flags                |
|----------------------|------------|----------------------|
| 8.5s                 | 51GB       |                      |
| 16.6s                | 30GB       | --offload_t5         |
| 20s                  | 24GB       | --offload_t5 --offload_transformer_during_vae|
| 10min                | 9GB        | --offload_t5 --offload_vae |

## Command Line Interface

```bash
python cli.py --input_dir /path/to/input/json/files --output_dir /path/to/output/directory --video --spz --ply
```

Parameters:
- `--input_dir`: Directory containing JSON files with generation parameters (required). We provide some examples in ```./examples``` that you can use the directory directly.
- `--output_dir`: Directory to save generated results (required)
- `--video`: Generate video output
- `--spz`: Export results in SPZ format
- `--ply`: Export results in PLY format
- `--video_fps`: Video frame rate (default: 15)
- some flags are shared with the Local Web Interface

**Note**: The CLI interface provides better rendering results compared to the web interface, as it uses uncompressed Gaussian Parameters. Use the CLI for comparison if you want to use FlashWorld as your baseline.

  
## More Generation Results

[https://github.com/user-attachments/assets/bbdbe5de-5e15-4471-b380-4d8191688d82](https://github.com/user-attachments/assets/53d41748-4c35-48c4-9771-f458421c0b38)


## Minimal example
If you encounter issues running the web interface and just want to check your set up, check out the [minimal example](/examples/minimal) to get started.

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

## Acknowledgements

This work is done with [Hunyuan-World Team](https://github.com/Tencent-Hunyuan/HunyuanWorld-1.0).

