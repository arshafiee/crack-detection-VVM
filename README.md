# Perceptual Crack Detection for Rendered 3D Textured Meshes
Software release of the "Perceptual Crack Detection for Rendered 3D Textured Meshes" QoMEX 2024 paper.

## Documentation
Please refer to our [published paper](https://arxiv.org/abs/2405.06143), where we thoroughly explain the proposed PCD method.

## Installation
```
cd path/to/this/repo
python3 -m venv venv
source venv/bin/activate
pip3 install -r requirements.txt
```

## Usage
To compute the raw quality score, the new quality score, and to generate a video highlighting the crack artifacts, you can run the following command with the following arguments:

`required arguments`:

* `--ref_add` reference video path, string,
* `--tst_add` test video path, string.

`optional arguments`:
* `-h` or `--help`: shows the help message and exits,
* `--method`: initial quality assessment method to be enhanced by the weight map, string, options: `lumaPSNR`, `SSIM`, `IW-SSIM`! (default: `IW-SSIM`),
* `--skip_fr`: number of consecutive frames to skip in each interval when generating the outputs, int, (default: `20`),
* `--out_fps`: desired frame rate (fps) for the output video, ignored when `--no_vid` flag is up, float, (default: `5.0`),
* `--no_vid`: flag to disable generating the output video,
* `--no_rqs`: flag to disable computing the raw quality score.

Please note that the height and width of the output video frames are two times the height and width of the input video frames.

The following `Example` operates on one out of every 5 consecutive frames of the reference and test videos to compute the raw and enhanced SSIM quality score and to generate a video with 5.0 frame rate that highlights the crack artifacts:
```
cd path_to_this_repo
python3 main.py \
--ref_add data/statue_Ref.mp4 \
--tst_add data/statue_simpL6_qp8_qt6_decompJPEG_2048x2048_Q90.mp4 \
--skip_fr 5 \
--method SSIM 
```

Output of the preceding example:

`Raw quality score: 0.9059885119062726`

`New quality score: 0.11615714537909715`

A sample frame of the generated video:

![Alt text](/img/sample_frame.png)

## Citation
If you use any part of this code or data in your research, please cite our [publication](https://arxiv.org/abs/2405.06143).

## Aknowledgement

The code for IW-SSIM and SSIM were adapted from the following references, respectively:

[1] PyTorch Image Quality: Metrics for Image Quality Assessment. Kastryulin, Sergey and Zakirov, Jamil and Prokopenko, Denis and Dylov, Dmitry V. arXiv 2022. https://arxiv.org/abs/2208.14818.

[2] https://github.com/mubeta06/python/blob/master/signal_processing/sp/ssim.py.

