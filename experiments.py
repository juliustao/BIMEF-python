import cv2
import BIMEF
import os
from scipy.io import loadmat

def enhance_img(filename):
    bgr_img = cv2.imread(filename, 1)
    b, g, r = cv2.split(bgr_img)  # get b,g,r
    rgb_img = cv2.merge([r, g, b])  # switch it to rgb

    # data_path = '/Users/carnitas/BIMEF/image.mat'
    # data = loadmat(data_path)
    # rgb_img = data['image']

    enhanced_rgb_img = BIMEF.BIMEF(rgb_img)
    r, g, b = cv2.split(enhanced_rgb_img)
    enhanced_bgr_img = cv2.merge([b, g, r])
    return enhanced_bgr_img

#
# if __name__ == '__main__':
#     rgb_img = loadmat('/Users/carnitas/BIMEF/newIint.mat')['I0']
#     enhanced_rgb_img = BIMEF.BIMEF(rgb_img)
#     r, g, b = cv2.split(enhanced_rgb_img)
#     enhanced_bgr_img = cv2.merge([b, g, r])
#     cv2.imwrite('test.jpg', enhanced_bgr_img)


if __name__ == '__main__':
    in_dir = '/Users/carnitas/Downloads/dark_images/'
    out_dir = '/Users/carnitas/Matlab2Python/enhanced_images/'
    print(in_dir)
    for in_filename in os.listdir(in_dir):
        print(in_filename)
        enhanced_bgr_img = enhance_img(os.path.join(in_dir, in_filename))
        out_filename = in_filename[0:in_filename.find('.')] + '_BIMEF_python.jpg'
        cv2.imwrite(os.path.join(out_dir, out_filename), enhanced_bgr_img)
    print(out_dir)
