# IMPORTANT: running this file is optional.
# Running it shall generate the files with "_rectified" suffix images
import cv2
import numpy as np


# reads image from path
def process_image(img_path):
    img = cv2.imread(img_path, 0)
    return img


# detects keypoints & descriptors via SIFT
def detect_kp_desc(img):
    sift = cv2.xfeatures2d.SIFT_create()
    kp, des = sift.detectAndCompute(img, None)
    return kp, des


# matches the descriptors via FLANN
def match_kp_desc(des1, des2):
    FLANN_INDEX_KDTREE = 1
    idx_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(idx_params, search_params)
    all_matches = flann.knnMatch(des1, des2, k=2)
    return all_matches


# finds 'good' matches using Euclidean distance
def filter_matches(matches, kp_left, kp_right):
    good_matches = []
    pts_left = []
    pts_right = []
    matches_pair = [[0, 0] for i in range(len(matches))]

    for i, (x, y) in enumerate(matches):
        if x.distance < 0.7 * y.distance:
            matches_pair[i] = [1, 0]
            good_matches.append(x)
            pts_right.append(kp_right[x.trainIdx].pt)
            pts_left.append(kp_left[x.queryIdx].pt)

    return pts_left, pts_right


# calculates the fundamental matrix using RANSAC
def calc_fundamental_matrix(pts_left, pts_right):
    pts_left = np.int32(pts_left)
    pts_right = np.int32(pts_right)
    fundamental_matrix, inliers = cv2.findFundamentalMat(pts_left, pts_right, cv2.FM_RANSAC)

    pts_left = pts_left[inliers.ravel() == 1]
    pts_right = pts_right[inliers.ravel() == 1]

    return fundamental_matrix, pts_left, pts_right

# draws epipolar lines on the images using the 'good' matches
def draw_epipolar_lines(img_left, img_right, lines, pts_left, pts_right):
    r, c = img_left.shape
    img_left_color = cv2.cvtColor(img_left, cv2.COLOR_GRAY2BGR)
    img_right_color = cv2.cvtColor(img_right, cv2.COLOR_GRAY2BGR)

    np.random.seed(0)

    for r, pt1, pt2 in zip(lines, pts_left, pts_right):
        color = tuple(np.random.randint(0, 255, 3).tolist())

        x1, y1 = map(int, [0, -r[2] / r[1]])
        x2, y2 = map(int, [c, -(r[2] + r[0] * c) / r[1]])

        img_left_color = cv2.line(img_left_color, (x1, y1), (x2, y2), color, 1)
        img_left_color = cv2.circle(img_left_color, tuple(pt1), 5, color, -1)
        img_right_color = cv2.circle(img_right_color, tuple(pt2), 5, color, -1)

    return img_left_color, img_right_color


# find epipolar lines using the fundamental matrix
def find_epipolar_lines(img_left, img_right, pts1, pts2, fundamental_matrix):
    lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1, 1, 2), 2, fundamental_matrix)
    lines1 = lines1.reshape(-1, 3)
    draw_epipolar_lines(img_left, img_right, lines1, pts1, pts2)

    lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1, 1, 2), 1, fundamental_matrix)
    lines2 = lines2.reshape(-1, 3)
    draw_epipolar_lines(img_right, img_left, lines2, pts2, pts1)


# rectifies and warps the uncalibrated images, and stores them
def rectify_images(img_left, img_right, pts1, pts2, fundamental_matrix, img_category):
    h1, w1 = img_left.shape
    h2, w2 = img_right.shape

    _, H1, H2 = cv2.stereoRectifyUncalibrated(
                    np.float32(pts1),
                    np.float32(pts2),
                    fundamental_matrix,
                    imgSize=(w1, h1), )

    img_left_rect = cv2.warpPerspective(img_left, H1, (w1, h1))
    img_right_rect = cv2.warpPerspective(img_right, H2, (w2, h2))

    cv2.imwrite(f"{img_category}_left_rectified.jpg", img_left)
    cv2.imwrite(f"{img_category}_right_rectified.jpg", img_right)


# IMPORTANT: instructions for running the code are as follows
if __name__ == "__main__":
    # original image paths go here
    img_left_path = "/Users/avimalhotra/Desktop/StereoMatchingTestings/Art/view1.png"
    img_right_path ="/Users/avimalhotra/Desktop/StereoMatchingTestings/Art/view5.png"

    img_left = process_image(img_left_path)
    img_right = process_image(img_right_path)

    kp1, des1 = detect_kp_desc(img_left)
    kp2, des2 = detect_kp_desc(img_right)

    matches = match_kp_desc(des1, des2)

    pts_left, pts_right = filter_matches(matches, kp1, kp2)

    fundamental_matrix, pts_left, pts_right = calc_fundamental_matrix(pts_left, pts_right)

    find_epipolar_lines(img_left, img_right, pts_left, pts_right, fundamental_matrix)

    # change the final string parameter as needed
    rectify_images(img_left, img_right, pts_left, pts_right, fundamental_matrix, "Art")


