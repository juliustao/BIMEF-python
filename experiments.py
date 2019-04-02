import cv2
import BIMEF

img_path = '/Users/carnitas/Downloads/dark_images/0cd9b7d9-36b5-4b73-a29d-5ed319eb007d.jpg'
bgr_img = cv2.imread(img_path, 1)
b, g, r = cv2.split(bgr_img)       # get b,g,r
rgb_img = cv2.merge([r, g, b])     # switch it to rgb

enhanced_img = BIMEF.BIMEF(rgb_img)
cv2.imwrite('/Users/carnitas/Matlab2Python/0cd9b7d9-36b5-4b73-a29d-5ed319eb007d_python_BIMEF.jpg', enhanced_img)
