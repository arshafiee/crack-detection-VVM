# Copyright (C) 2023  Armin Shafiee Sarvestani
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
# Please email a5shafie@uwaterloo.ca for inquiries.

import cv2


class InvalidVideoAttributeError(Exception):
    def __init__(self, attribute_name, reason):
        self.attribute_name = attribute_name
        self.reason = reason
        super().__init__(f"Invalid video attribute '{attribute_name}': {reason}")


def check_videos_atts(cap_ref, cap_tst):

    if not cap_ref.isOpened():
        raise FileNotFoundError("Error opening reference video stream.")
    if not cap_tst.isOpened():
        raise FileNotFoundError("Error opening test video stream.")

    # resolution
    ref_resolution = int(cap_ref.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap_ref.get(cv2.CAP_PROP_FRAME_HEIGHT))
    tst_resolution = int(cap_tst.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap_tst.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if not ref_resolution == tst_resolution:
        raise InvalidVideoAttributeError('resolution', f'resolutions of reference {ref_resolution} and test'
                                                       f' {tst_resolution} videos do not match!')

    # frame rate
    ref_fps = cap_ref.get(cv2.CAP_PROP_FPS)
    tst_fps = cap_tst.get(cv2.CAP_PROP_FPS)
    if not ref_fps == tst_fps:
        raise InvalidVideoAttributeError('frame rate', f'frame rates of reference ({ref_fps}) and test ({tst_fps})'
                                                       f' videos do not match!')

    # num frames
    ref_fnum = int(cap_ref.get(cv2.CAP_PROP_FRAME_COUNT))
    tst_fnum = int(cap_tst.get(cv2.CAP_PROP_FRAME_COUNT))
    if not ref_fnum == tst_fnum:
        raise InvalidVideoAttributeError('number of frames',
                                         f'number of frames of reference ({ref_fnum}) and test ({tst_fnum}) videos'
                                         f' do not match!')

    # video format
    ref_fourcc = int(cap_ref.get(cv2.CAP_PROP_FOURCC))
    ref_fourcc = chr(ref_fourcc & 0xff) + chr((ref_fourcc >> 8) & 0xff) + chr((ref_fourcc >> 16) & 0xff) + chr(
        (ref_fourcc >> 24) & 0xff)
    tst_fourcc = int(cap_tst.get(cv2.CAP_PROP_FOURCC))
    tst_fourcc = chr(tst_fourcc & 0xff) + chr((tst_fourcc >> 8) & 0xff) + chr((tst_fourcc >> 16) & 0xff) + chr(
        (tst_fourcc >> 24) & 0xff)
    if not ref_fourcc == tst_fourcc:
        raise InvalidVideoAttributeError('video format',
                                         f'formats of reference ({ref_fourcc}) and test ({tst_fourcc}) videos'
                                         f' do not match!')


def check_frames_atts(ref_frame, tst_frame):

    # num channels
    ref_nc = ref_frame.shape[2] if len(ref_frame.shape) == 3 else 1
    tst_nc = tst_frame.shape[2] if len(tst_frame.shape) == 3 else 1
    if not ref_nc == tst_nc:
        raise InvalidVideoAttributeError('number of channels',
                                         f'number of channels of reference ({ref_nc}) and test ({tst_nc}) videos'
                                         f' do not match!')

    # bit depth
    ref_bd = ref_frame.dtype
    tst_bd = tst_frame.dtype
    if not ref_bd == tst_bd:
        raise InvalidVideoAttributeError('bit depth',
                                         f'bit depth of reference ({ref_bd}) and test ({tst_bd}) videos'
                                         f' do not match!')
