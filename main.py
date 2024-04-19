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

import argparse

from src.weighted_quality import WeightedQuality

if __name__ == '__main__':

    parser = argparse.ArgumentParser("main", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    required = parser.add_argument_group('required arguments')
    required.add_argument('--ref_add', type=str, help='reference video path!', required=True)
    required.add_argument('--tst_add', type=str, help='test video path!', required=True)
    parser.add_argument('--method', type=str, default='IW-SSIM',
                        help='initial quality assessment method to be enhanced by the weight map. Options: '
                             '"lumaPSNR", "SSIM", "IW-SSIM"!')
    parser.add_argument('--skip_fr', type=int, default=20,
                        help='number of consecutive frames to skip in each interval when generating the outputs!')
    parser.add_argument('--out_fps', type=float, default=5.0,
                        help='desired frame rate (fps) for the output video! ignored when "--no_vid" flag is up')
    parser.add_argument('--no_vid', action='store_true', help='flag to disable generating the output video!')
    parser.add_argument('--no_rqs', action='store_true', help='flag to disable computing the raw quality score!')
    args = parser.parse_args()

    quality_class = WeightedQuality(method=args.method, skip_fr=args.skip_fr, out_fps=args.out_fps, no_vid=args.no_vid,
                                    no_rqs=args.no_rqs)
    quality_class.test(ref_add=args.ref_add, tst_add=args.tst_add)
