import cv2
import BIMEF
from scipy.io import loadmat

img_path = '/Users/carnitas/BIMEF/data/TEST/0cd9b7d9-36b5-4b73-a29d-5ed319eb007d.jpg'

bgr_img = cv2.imread(img_path, 1)
b, g, r = cv2.split(bgr_img)       # get b,g,r
rgb_img = cv2.merge([r, g, b])     # switch it to rgb

# data_path = '/Users/carnitas/BIMEF/image.mat'
#
# data = loadmat(data_path)
#
# rgb_img = data['image']

enhanced_rgb_img = BIMEF.BIMEF(rgb_img)
r, g, b = cv2.split(enhanced_rgb_img)
enhanced_bgr_img = cv2.merge([b, g, r])

cv2.imwrite('/Users/carnitas/Matlab2Python/0cd9b7d9-36b5-4b73-a29d-5ed319eb007d_python_BIMEF.jpg', enhanced_bgr_img)
