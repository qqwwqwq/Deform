import cv2
import matplotlib.pyplot as plt

# 查找棋盘格 角点
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)

# 棋盘格参数
corners_vertical = 5   # 纵向角点个数;
corners_horizontal = 9  # 纵向角点个数;
pattern_size = (corners_vertical, corners_horizontal)

#
# def find_corners_sb(img):
#     """
#     查找棋盘格角点函数 SB升级款
#     :param img: 处理原图
#     """
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     # 查找棋盘格角点;
#     ret, corners = cv2.findChessboardCornersSB(gray, pattern_size, cv2.CALIB_CB_EXHAUSTIVE + cv2.CALIB_CB_ACCURACY)
#     if ret:
#         # 显示角点
#         cv2.drawChessboardCorners(img, pattern_size, corners, ret)


def find_corners(img):
    """
    查找棋盘格角点函数
    :param img: 处理原图
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 查找棋盘格角点;
    ret, corners = cv2.findChessboardCorners(gray, pattern_size, cv2.CALIB_CB_ADAPTIVE_THRESH +
                                             cv2.CALIB_CB_FAST_CHECK +
                                             cv2.CALIB_CB_FILTER_QUADS)
    if ret:
        # 精细查找角点
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        print(corners2[22][0][0])
        cv2.circle(img, (corners2[22][0][0], corners2[22][0][1]), 60, (0, 0, 255), 5)
        cv2.circle(img, (corners2[21][0][0], corners2[21][0][1]), 60, (0, 0, 255), 5)
        cv2.circle(img, (corners2[23][0][0], corners2[23][0][1]), 60, (0, 0, 255), 5)
        cv2.circle(img, (corners2[22][0][0], corners2[22][0][1]), 1, (255, 0, 0), 10)
        cv2.circle(img, (corners2[21][0][0], corners2[21][0][1]), 1, (255, 0, 0), 10) # 显示角点
        cv2.circle(img, (corners2[23][0][0], corners2[23][0][1]), 1, (255, 0, 0), 10) #cv2.drawChessboardCorners(img, pattern_size, corners2, ret)


def main():
    #img_src = cv2.imread("/Users/apple/Downloads/image/left.jpg")
    img_src = cv2.imread("/Users/apple/Desktop/IMG_2192.JPG",-1)
    plt.figure("Image")
    find_corners(img_src)
    plt.imshow(img_src)
    plt.show()


if __name__ == '__main__':
    main()