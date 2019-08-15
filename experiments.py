import cv2
from BIMEF import BIMEF
import os
import time

def enhance_img(filename):
    bgr_img = cv2.imread(filename, 1)
    b, g, r = cv2.split(bgr_img)  # get b,g,r
    rgb_img = cv2.merge([r, g, b])  # switch it to rgb

    start = time.time()
    enhanced_rgb_img = BIMEF(rgb_img)
    end = time.time()
    run_time = end - start
    print('BIMEF run time: ' + str(run_time) + '\n')

    r, g, b = cv2.split(enhanced_rgb_img)
    enhanced_bgr_img = cv2.merge([b, g, r])
    return enhanced_bgr_img


if __name__ == '__main__':
    in_dir = '/Users/carnitas/dark_images/'
    assert(os.path.isdir(in_dir), 'The given input directory does not exist.')
    out_dir = '/Users/carnitas/Matlab2Python/cv2_resize_linear/'
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)
    print('Dark images pulled from: ' + in_dir)
    start = time.time()
    for in_filename in os.listdir(in_dir):
        print(in_filename)
        enhanced_bgr_img = enhance_img(os.path.join(in_dir, in_filename))
        period_index = in_filename.find('.')
        out_filename = in_filename[:period_index] + '_BIMEF_python' + in_filename[period_index:]
        cv2.imwrite(os.path.join(out_dir, out_filename), enhanced_bgr_img)
    end = time.time()
    run_time = end - start
    print('Total run time: ' + str(run_time))
    print('Enhanced images saved to: ' + out_dir)
