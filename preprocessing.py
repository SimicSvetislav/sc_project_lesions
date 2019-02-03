import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.measure import compare_ssim

def preprocess(img):
    
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))

    img_dilated = cv2.dilate(img_gray, kernel, iterations=5)
    img_eroded = cv2.erode(img_dilated, kernel, iterations=5)
    
    (score, img_diff) = compare_ssim(img_eroded, img_gray, full=True)
    
    img_diff = (img_diff * 255).astype("uint8")
    print(img_diff.dtype)
    
    mask_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))

    img_dilated_mask = cv2.dilate(img_diff, mask_kernel, iterations=3)
    img_eroded_mask = cv2.erode(img_dilated_mask, mask_kernel, iterations=3)
    
    ret, img_bin_mask = cv2.threshold(img_eroded_mask, 0, 255, cv2.THRESH_OTSU)
    img_bin_mask_inv = cv2.bitwise_not(img_bin_mask)
    
    img_copy_gray = np.copy(img_gray)
    img_copy_gray[img_bin_mask==0] = 0
    
    img_ip_gray = cv2.inpaint(img_copy_gray,img_bin_mask_inv,3,cv2.INPAINT_TELEA)
    
    ksize = 5

    img_final_g = cv2.medianBlur(img_ip_gray, ksize)
    
    # Abonded
    #img_final_eq = cv2.equalizeHist(img_final_g)
    
    #cv2.imwrite('pp_output/1.jpg', img_final_g)
    
    #res = np.hstack((img_final_g,img_final_eq))
    #plt.imshow(res, 'gray')
    
    return img_final_g    

def preprocessString(image):
    img = cv2.imread(image)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    return preprocess(img)
    
#out = preprocessString('images/ISIC_0024799.jpg')
#plt.imshow(out, 'gray')
    