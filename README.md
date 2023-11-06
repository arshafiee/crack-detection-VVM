# Crack Detection for Objective Quality Assessment of Rendered 3D Textured Meshes

## Usage: main.py
To compute the raw quality score, the new quality score, and to generate a video highlighting the geometric defects, you can run the following piece of code with the following arguments.

`required arguments`:

* `--ref_add` reference video path, string,
* `--tst_add` test video path, string,

`optional arguments`:
* `-h` or `--help`: shows the help message and exits,
* `--method`: initial quality assessment method to be enhanced by the weight map, string, options: `lumaPSNR`, `SSIM`, `IW-SSIM`! (default: `IW-SSIM`),
* `--skip_fr`: number of consecutive frames to skip in each interval when generating the outputs, int, (default: `20`),
* `--out_fps`: desired frame rate (fps) for the output video, ignored when `--no_vid` flag is up, float, (default: `5.0`),
* `--no_vid`: flag to disable generating the output video,
* `--no_rqs`: flag to disable computing the raw quality score.

Please note that the height and width of the output video frames are two times the height and width of the input video frames.

The following `Example` operates on one out of every 30 consecutive frames of the reference and test videos to compute the raw and enhanced SSIM quality score and to generate a video with 5.0 frame rate that highlights the geometric defects:
```
cd path_to_this_repo
python3 src/main.py \
--ref_add data/orbiter_space_shutter_C0-L5_qp0_qt0.mp4 \
--tst_add data/orbiter_space_shutter_dec0.20_qp9_qt8_cqlevel63.mp4 \
--skip_fr 30 \
--method SSIM 
```

Output of the preceding example:

`Raw quality score: 0.9470075559485853`

`New quality score: 0.12092643017352467`

A sample frame of the generated video:

![Alt text](/img/sample_frame.png)

`Aknowledgement`:

The code for IW-SSIM and SSIM were adapted from the following references, respectively:

[1] PyTorch Image Quality: Metrics for Image Quality Assessment. Kastryulin, Sergey and Zakirov, Jamil and Prokopenko, Denis and Dylov, Dmitry V. arXiv 2022. https://arxiv.org/abs/2208.14818.

[2] https://github.com/mubeta06/python/blob/master/signal_processing/sp/ssim.py.

