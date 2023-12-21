import cv2
import numpy as np
import os
import random

def draw_epilines(img1, img2, pts1, pts2, F):
    # 计算epilines。 结果是 ax + by + c = 0 的形式
    lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1, 1, 2), 2, F)
    lines1 = lines1.reshape(-1, 3)
    img1_epilines = _drawlines(img1, lines1, pts1, pts2)

    lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1, 1, 2), 1, F)
    lines2 = lines2.reshape(-1, 3)
    img2_epilines = _drawlines(img2, lines2, pts2, pts1)

    return img1_epilines, img2_epilines

def _drawlines(img, lines, pts1, pts2):
    r, c = img.shape
    for r, pt1, pt2 in zip(lines, pts1, pts2):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0, y0 = map(int, [0, -r[2] / r[1]])
        x1, y1 = map(int, [c, -(r[2] + r[0] * c) / r[1]])
        img = cv2.line(img, (x0, y0), (x1, y1), color, 1)
        img = cv2.circle(img, tuple(pt1), 5, color, -1)
    return img

def find_matches(img1, img2, ratio_threshold=0.7):
    # 使用SIFT特征检测和描述子提取
    sift = cv2.SIFT_create()

    # 在两个图像中检测关键点和计算描述子
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # 使用BFMatcher进行描述子匹配
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    matches = bf.knnMatch(des1, des2, k=2)

    # 通过ratio test筛选出好的匹配
    good_matches = []
    for m, n in matches:
        if m.distance < ratio_threshold * n.distance:
            good_matches.append(m)

    # 获取匹配的点坐标
    pts1 = np.int32([kp1[m.queryIdx].pt for m in good_matches])
    pts2 = np.int32([kp2[m.trainIdx].pt for m in good_matches])

    return pts1, pts2

# fisheye_folder = r"E:\SkdPaper\STAR_Center\result\titan\titan_0"
# undistorted_folder = r"E:\SkdPaper\STAR_Center\result\titan\titan_0_READ_workspace\undistorted_images_0748x1328_0.8"
# file_name = "1693123178400000000_3.jpg"

# fisheye_file = os.path.join(fisheye_folder, file_name)
# undistorted_file = os.path.join(undistorted_folder, file_name)

# fisheye_img = cv2.imread(fisheye_file, cv2.IMREAD_GRAYSCALE)
# undistorted_img = cv2.imread(undistorted_file, cv2.IMREAD_GRAYSCALE)

# print(fisheye_img.shape, undistorted_img.shape)

# pts1, pts2 = find_matches(fisheye_img, undistorted_img)
# print(pts1, pts2)
# F, _ = cv2.findFundamentalMat(pts1, pts2)
# fisheye_result1, undistorted_result1 = draw_epilines(fisheye_img, undistorted_img, pts1, pts2, F)

# os.makedirs("/tests/results/epipolar_line/")
# cv2.imwrite("/tests/results/epipolar_line/fisheye.jpg", fisheye_result1)
# cv2.imwrite("/tests/results/epipolar_line/undistorted.jpg", undistorted_result1)


undistorted_folder = r"E:\SkdPaper\STAR_Center\result\titan\titan_0_READ_workspace\undistorted_images_0748x1328_0.8"
file_name_1 = "1693123147400000000_3.jpg"
file_name_2 = "1693123149800000000_3.jpg"

file_1 = os.path.join(undistorted_folder, file_name_1)
file_2 = os.path.join(undistorted_folder, file_name_2)

img1 = cv2.imread(file_1, cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread(file_2, cv2.IMREAD_GRAYSCALE)

pts1, pts2 = find_matches(img1, img2)
print(len(pts1), len(pts2))
indexes = random.sample(range(0, len(pts1)), 15)
pts1_sele = []
pts2_sele = []
for index in indexes:
    pts1_sele.append(pts1[index])
    pts2_sele.append(pts2[index])
pts1_sele = np.array(pts1_sele)
pts2_sele = np.array(pts2_sele)

print(pts1_sele, pts2_sele)
F, _ = cv2.findFundamentalMat(pts1_sele, pts2_sele)
result1, result2 = draw_epilines(img1, img2, pts1_sele, pts2_sele, F)

os.makedirs("./tests/results/epipolar_line/", exist_ok=True)
cv2.imwrite("./tests/results/epipolar_line/un_res1.jpg", result1)
cv2.imwrite("./tests/results/epipolar_line/un_res2.jpg", result2)