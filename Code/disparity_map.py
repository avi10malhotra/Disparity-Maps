import cv2
import numpy as np


# builds disparity map from the rectified images
def calc_map(img_left, img_right):

    # StereoSGBM parameters (a lot of hit and trial to get the best results)
    window_size = 3
    left_matcher = cv2.StereoSGBM_create(
        minDisparity=-1,
        numDisparities=80,

        blockSize=window_size,
        P1=8 * 3 * window_size,
        P2=32 * 3 * window_size,

        disp12MaxDiff=12,
        uniquenessRatio=10,
        speckleWindowSize=50,
        speckleRange=30,
        preFilterCap=60,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY )

    right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)

    # WLS filter parameters (again, hit and trial was needed)
    lambda_val = 80000
    sigma_val = 1.2

    # post-filtering to remove noise and refine the disparity map
    wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
    wls_filter.setLambda(lambda_val)
    wls_filter.setSigmaColor(sigma_val)

    disp_left = left_matcher.compute(img_left, img_right)
    disp_right = right_matcher.compute(img_right, img_left)
    disp_left = np.int16(disp_left)
    disp_right = np.int16(disp_right)

    filtered_img = wls_filter.filter(disp_left, img_left, None, disp_right)
    filtered_img = cv2.normalize(src=filtered_img, dst=filtered_img, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX);
    filtered_img = np.uint8(filtered_img)

    return filtered_img

# calculates psnr as per given criteria
def calc_psnr(disparity_map, ground_truth):
    mse = np.mean((disparity_map - ground_truth) ** 2)

    if mse == 0:
        return float('inf')
    return 20 * np.log10(255 ** 2 / mse)

# IMPORTANT: instructions for running the code are as follows
if __name__ == "__main__":
    # Enter the path to RECTIFIED images (see "stereo_rectification.py")
    # OR enter the direct file path to the images
    img_left_rectified_path = "/Users/avimalhotra/Desktop/lastattempt/Art_left_rectified.jpg"
    img_right_rectified_path = "/Users/avimalhotra/Desktop/lastattempt/Art_left_rectified.jpg"
    ground_truth_path = "/Users/avimalhotra/Desktop/StereoMatchingTestings/Art/disp1.png"

    img_left = cv2.imread(img_left_rectified_path, 0)
    img_right = cv2.imread(img_right_rectified_path, 0)
    img_ground_truth = cv2.imread(ground_truth_path, 0)

    disparity_map = calc_map(img_left, img_right)

    # Enter the path to save the disparity map here
    img_category = "Art"
    cv2.imwrite(f"/Users/avimalhotra/Desktop/lastattempt/{img_category}_disparity_map.png", disparity_map)
    cv2.imshow(f"{img_category}_disparity_map", disparity_map)

    # reference PSNR values for the images are as follows:
    # Art = 56.049
    # Dolls = 55.777
    # Reindeer = 55.482
    psnr = calc_psnr(disparity_map, img_ground_truth)
    print(f"PSNR for {img_category} category: {round(psnr, 3)}")

