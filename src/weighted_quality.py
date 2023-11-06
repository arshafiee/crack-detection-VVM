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
import torch
import numpy as np
from scipy.signal.windows import gaussian
from scipy import stats
from ssim import ssim
from cbiw_ssim import cb_information_weighted_ssim
from utils import check_videos_atts, check_frames_atts


class WeightedQuality:

    def __init__(self, method, skip_fr, out_fps, no_vid, no_rqs):

        self.method = method
        self.skip_fr = skip_fr
        self.out_fps = out_fps
        self.no_vid = no_vid
        self.no_rqs = no_rqs
        if not no_vid:
            self.video_file = None

        # constants
        self._C1 = 0.0001
        self._T1 = 0.1
        self._C = 0.01
        self._gaussian_size = 5
        self._gaussian_std = 1.5
        self._sigmoid_threshold = 0.25
        self._morph_kernel = np.ones((5, 5))

    def _find_contrast_map(self, img):

        img = np.double(img)

        gauss_1d = gaussian(self._gaussian_size, self._gaussian_std)
        raw_window = np.outer(gauss_1d, gauss_1d)
        window = raw_window / np.sum(raw_window)

        mu = cv2.filter2D(img, -1, window)
        mu_sq = mu * mu
        sigma_sq = cv2.filter2D(img * img, -1, window) - mu_sq
        sigma_sq[sigma_sq < 0] = 0
        sigma = np.sqrt(sigma_sq)

        return sigma

    def _apply_piecewise_sigmoid(self, img, threshold):

        result_img = img.copy()
        sigmoid_img = 1 / (1 + np.exp(-1 / threshold * (img - threshold)))
        result_img[img <= threshold] = 0.0
        result_img[img > threshold] = sigmoid_img[img > threshold]

        return result_img

    def _find_crack_map(self, raw_frame_ref, raw_frame_dis):

        frame_ref = cv2.cvtColor(raw_frame_ref, cv2.COLOR_BGR2GRAY)
        db_frame_ref = np.double(frame_ref) / 255.0
        frame_dis = cv2.cvtColor(raw_frame_dis, cv2.COLOR_BGR2GRAY)
        db_frame_dis = np.double(frame_dis) / 255.0

        _, db_frame_dif = cv2.threshold(np.abs(db_frame_ref - db_frame_dis), self._T1, 1.0, cv2.THRESH_TOZERO)
        db_frame_dis_lap = cv2.Laplacian(db_frame_dis, ddepth=-1, ksize=5)

        db_frame_ref_cont = self._find_contrast_map(frame_ref)

        db_frame_crack = db_frame_dif / (db_frame_ref_cont + self._C)
        db_frame_crack_lap = db_frame_crack * np.abs(db_frame_dis_lap)
        db_frame_final = self._apply_piecewise_sigmoid(db_frame_crack_lap, threshold=self._sigmoid_threshold)
        db_eff_crack_map = (1 + self._C1) / (self._C1 + 1 - db_frame_final)

        return db_frame_final, db_eff_crack_map

    def _generate_video_frame(self, ref_frame, tst_frame, crack_mask):

        # dilating the mask for better visualization
        crack_mask = cv2.dilate(crack_mask, np.ones((3, 3)))
        crack_mask = np.round(crack_mask * 255.0).astype(np.uint8)
        gray_crack_mask = cv2.cvtColor(crack_mask, cv2.COLOR_GRAY2RGB)
        gray_tst_frame = cv2.cvtColor(cv2.cvtColor(tst_frame, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2RGB)
        gray_ref_frame = cv2.cvtColor(cv2.cvtColor(ref_frame, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2RGB)

        frame_with_cracks = gray_tst_frame.copy()
        row_ind, col_ind = np.where(crack_mask > 0)
        frame_with_cracks[row_ind, col_ind, 0] = crack_mask[row_ind, col_ind]  # paint cracks in red
        frame_with_cracks[row_ind, col_ind, 1:] = [0, 0]

        # putting text on the frames
        half_width = gray_ref_frame.shape[1] // 2
        cv2.putText(gray_tst_frame, 'Test frame', (half_width, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                    cv2.LINE_AA)
        cv2.putText(gray_ref_frame, 'Reference frame', (half_width, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255),
                    2, cv2.LINE_AA)
        cv2.putText(frame_with_cracks, 'Test frame + highlighted defects', (half_width, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(gray_crack_mask, 'highlighted defects', (half_width, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (255, 255, 255), 2, cv2.LINE_AA)

        lower_hf_frame = np.concatenate((gray_crack_mask, frame_with_cracks), axis=1)
        upper_hf_frame = np.concatenate((gray_ref_frame, gray_tst_frame), axis=1)
        new_frame = np.concatenate((upper_hf_frame, lower_hf_frame), axis=0)
        self.video_file.write(cv2.cvtColor(new_frame, cv2.COLOR_RGB2BGR))

    def _background_floor_mask(self, ref_img, tst_img, color):

        background_img = np.ones_like(ref_img, dtype=np.uint8) * color

        dif_frame_ref = abs(ref_img - background_img)
        _, mask_ref = cv2.threshold(dif_frame_ref, 0, 1.0, cv2.THRESH_BINARY)
        mask_ref = cv2.morphologyEx(mask_ref, cv2.MORPH_OPEN, self._morph_kernel)
        mask_ref = cv2.morphologyEx(mask_ref, cv2.MORPH_CLOSE, self._morph_kernel)

        tst_frame_dis = abs(tst_img - background_img)
        _, mask_tst = cv2.threshold(tst_frame_dis, 0, 1.0, cv2.THRESH_BINARY)
        mask_tst = cv2.morphologyEx(mask_tst, cv2.MORPH_OPEN, self._morph_kernel)
        mask_tst = cv2.morphologyEx(mask_tst, cv2.MORPH_CLOSE, self._morph_kernel)

        mask = cv2.bitwise_or(mask_tst, mask_ref)

        return mask

    def _crop_bounding_box(self, ref_img, tst_img):

        ref_img = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
        tst_img = cv2.cvtColor(tst_img, cv2.COLOR_BGR2GRAY)

        back_freq, floor_freq = 0.0, 0.0

        if (ref_img[:50, :50] == ref_img[0, 0]).all():
            back_color, back_freq = stats.mode(ref_img.flatten(), keepdims=False)
            back_freq /= len(ref_img.flatten())
            floor_color, floor_freq = stats.mode(ref_img[ref_img != back_color].flatten(), keepdims=False)
            floor_freq /= len(ref_img[ref_img != back_color].flatten())

        if back_freq > 0.4 and floor_freq > 0.25:

            background_mask = self._background_floor_mask(ref_img, tst_img, color=back_color)
            floor_mask = self._background_floor_mask(ref_img, tst_img, color=floor_color)
            mask = cv2.bitwise_and(background_mask, floor_mask)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._morph_kernel)
            x, y, width, height = cv2.boundingRect(cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                                                                    np.ones((11, 11))))
            x1 = x
            y1 = y
            x2 = x1 + width
            y2 = y1 + height

        else:

            ref_canny_edge = cv2.Canny(ref_img, 50, 150)
            x_ref, y_ref, width_ref, height_ref = cv2.boundingRect(ref_canny_edge)

            tst_canny_edge = cv2.Canny(tst_img, 50, 150)
            x_tst, y_tst, width_tst, height_tst = cv2.boundingRect(tst_canny_edge)

            x1, y1 = min(x_ref, x_tst), min(y_ref, y_tst)
            x2, y2 = max(x_ref + width_ref, x_tst + width_tst), max(y_ref + height_ref, y_tst + height_tst)

        # IW-SSIM input should be larger than 161x161
        if x2 - x1 < 161:
            dif = 161 - (x2 - x1)
            x1, x2 = x1 - int(np.ceil(dif / 2)), x2 + int(np.ceil(dif / 2))
        if y2 - y1 < 161:
            dif = 161 - (y2 - y1)
            y1, y2 = y1 - int(np.ceil(dif / 2)), y2 + int(np.ceil(dif / 2))

        return x1, x2, y1, y2

    def _compute_enhanced_quality_score(self, ref_frame, tst_frame, crp_coords, eff_crack_map):

        raw_quality_score = None

        if self.method == 'lumaPSNR':

            ref_y = np.double(cv2.cvtColor(ref_frame, cv2.COLOR_BGR2YUV)[:, :, 0])
            tst_y = np.double(cv2.cvtColor(tst_frame, cv2.COLOR_BGR2YUV)[:, :, 0])
            squared_err = (ref_y - tst_y) ** 2
            if not self.no_rqs:
                raw_quality_score = 10 * np.log10((255 ** 2) / np.mean(squared_err))

            crp_squared_err = squared_err[crp_coords[2]:crp_coords[3], crp_coords[0]:crp_coords[1]]
            crack_enhanced_mse = np.sum(crp_squared_err * eff_crack_map / np.sum(eff_crack_map))
            enh_quality_score = 10 * np.log10((255 ** 2) / crack_enhanced_mse)

        elif self.method == 'SSIM':

            ref_frame = cv2.cvtColor(ref_frame, cv2.COLOR_BGR2GRAY)
            tst_frame = cv2.cvtColor(tst_frame, cv2.COLOR_BGR2GRAY)
            if not self.no_rqs:
                ssim_map = ssim(ref_frame, tst_frame)
                raw_quality_score = np.mean(ssim_map)

            crp_ref_frame = ref_frame[crp_coords[2]:crp_coords[3], crp_coords[0]:crp_coords[1]]
            crp_tst_frame = tst_frame[crp_coords[2]:crp_coords[3], crp_coords[0]:crp_coords[1]]
            ssim_map = ssim(crp_ref_frame, crp_tst_frame)
            pad_y = (crp_ref_frame.shape[0] - ssim_map.shape[0]) // 2
            pad_x = (crp_ref_frame.shape[1] - ssim_map.shape[1]) // 2
            crack_enhanced_ssim = ssim_map * eff_crack_map[pad_y:-pad_y, pad_x:-pad_x] \
                                  / np.sum(eff_crack_map[pad_y:-pad_y, pad_x:-pad_x])
            enh_quality_score = np.sum(crack_enhanced_ssim)

        elif self.method == 'IW-SSIM':

            ref_img = cv2.cvtColor(ref_frame, cv2.COLOR_BGR2RGB)
            tst_img = cv2.cvtColor(tst_frame, cv2.COLOR_BGR2RGB)
            tensor_ref_y = torch.tensor(ref_img).permute(2, 0, 1).unsqueeze(dim=0)
            tensor_tst_y = torch.tensor(tst_img).permute(2, 0, 1).unsqueeze(dim=0)
            if not self.no_rqs:
                raw_quality_score = cb_information_weighted_ssim(x=tensor_ref_y, y=tensor_tst_y, eff_crack_map=None,
                                                                 data_range=255)

            crp_tensor_ref_y = tensor_ref_y[:, :, crp_coords[2]:crp_coords[3], crp_coords[0]:crp_coords[1]]
            crp_tensor_tst_y = tensor_tst_y[:, :, crp_coords[2]:crp_coords[3], crp_coords[0]:crp_coords[1]]
            tensor_eff_crack_map = torch.tensor(eff_crack_map,
                                                dtype=torch.float32).unsqueeze(dim=0).unsqueeze(dim=0)
            enh_quality_score = cb_information_weighted_ssim(x=crp_tensor_ref_y, y=crp_tensor_tst_y,
                                                             eff_crack_map=tensor_eff_crack_map,
                                                             data_range=255)

        else:
            raise ValueError(
                f'Invalid initial quality assessment method: "{self.method}"! Should be "lumaPSNR", or '
                f'"SSIM", or "IW-SSIM"!')

        return raw_quality_score, enh_quality_score

    def test(self, ref_add, tst_add):

        cap_ref = cv2.VideoCapture(ref_add)
        cap_tst = cv2.VideoCapture(tst_add)

        check_videos_atts(cap_ref=cap_ref, cap_tst=cap_tst)

        if not self.no_vid:
            width = int(cap_ref.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap_ref.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.video_file = cv2.VideoWriter(tst_add.split('/')[-1][:-4] + "_output.mp4",
                                              cv2.VideoWriter_fourcc('m', 'p', '4', 'v'),
                                              self.out_fps,
                                              (2 * width, 2 * height))

        frame_counter = 0
        enhc_quality_values = []
        raw_quality_values = []

        while cap_ref.isOpened() or cap_tst.isOpened():

            ret_ref, ref_frame = cap_ref.read()
            tst_dis, tst_frame = cap_tst.read()

            if ret_ref and tst_dis:

                if frame_counter == 0:
                    check_frames_atts(ref_frame=ref_frame, tst_frame=tst_frame)

                frame_counter += 1

                if frame_counter % self.skip_fr == 0:

                    if not self.no_vid:
                        crack_mask, eff_crack_map = self._find_crack_map(ref_frame, tst_frame)
                        self._generate_video_frame(ref_frame=ref_frame, tst_frame=tst_frame, crack_mask=crack_mask)
                        crp_coords = self._crop_bounding_box(ref_frame, tst_frame)
                        eff_crack_map = eff_crack_map[crp_coords[2]:crp_coords[3], crp_coords[0]:crp_coords[1]]

                    else:
                        crp_coords = self._crop_bounding_box(ref_frame, tst_frame)
                        crack_mask, eff_crack_map = self._find_crack_map(ref_frame[crp_coords[2]:crp_coords[3],
                                                                         crp_coords[0]:crp_coords[1]],
                                                                         tst_frame[crp_coords[2]:crp_coords[3],
                                                                         crp_coords[0]:crp_coords[1]])
                    raw_quality_value, enhanced_quality_value = self._compute_enhanced_quality_score(ref_frame,
                                                                                                     tst_frame,
                                                                                                     crp_coords,
                                                                                                     eff_crack_map)
                    enhc_quality_values.append(enhanced_quality_value)
                    raw_quality_values.append(raw_quality_value)
            else:
                break

        cap_ref.release()
        cap_tst.release()
        if not self.no_vid:
            self.video_file.release()

        raw_quality_score = None
        enhanced_quality_score = np.mean(enhc_quality_values)
        if not self.no_rqs:
            raw_quality_score = np.mean(raw_quality_values)
            print(f'Raw quality score: {raw_quality_score}')
        print(f'New quality score: {enhanced_quality_score}')
        return raw_quality_score, enhanced_quality_score
