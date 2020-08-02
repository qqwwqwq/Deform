import cv2
import numpy as np


def drawMatches(img1, kp1, img2, kp2, matches):
    """
    My own implementation of cv2.drawMatches as OpenCV 2.4.9
    does not have this function available but it's supported in
    OpenCV 3.0.0

    This function takes in two images with their associated
    keypoints, as well as a list of DMatch data structure (matches)
    that contains which keypoints matched in which images.

    An image will be produced where a montage is shown with
    the first image followed by the second image beside it.

    Keypoints are delineated with circles, while lines are connected
    between matching keypoints.

    img1,img2 - Grayscale images
    kp1,kp2 - Detected list of keypoints through any of the OpenCV keypoint
              detection algorithms
    matches - A list of matches of corresponding keypoints through any
              OpenCV keypoint matching algorithm
    """

    # Create a new output image that concatenates the two images together
    # (a.k.a) a montage
    rows1 = img1.shape[0]
    cols1 = img1.shape[1]
    rows2 = img2.shape[0]
    cols2 = img2.shape[1]

    out = np.zeros((max([rows1,rows2]),cols1+cols2,3))

    # Place the first image to the left
    out[:rows1,:cols1] = np.dstack([img1])

    # Place the next image to the right of it
    out[:rows2,cols1:] = np.dstack([img2])

    # For each pair of points we have between both images
    # draw circles, then connect a line between them
    z=[(0,0,255),(0,255,0),(255,0,0),(100,30,200),(111,200,255),(122,30,0),(255,60,255),(125,125,0)]
    for i in range(len(matches)):
        q=z[i]
        for mat in matches[i]:
            # Get the matching keypoints for each of the images
            img1_idx = mat[0].queryIdx
            img2_idx = mat[0].trainIdx

            # x - columns
            # y - rows
            (x1, y1) = kp1[img1_idx].pt
            (x2, y2) = kp2[img2_idx].pt

            # Draw a small circle at both co-ordinates
            # radius 4
            # colour blue
            # thickness = 1


            cv2.circle(out, (int(np.round(x1)), int(np.round(y1))), 2, q, 1)  # 画圆，cv2.circle()参考官方文档
            cv2.circle(out, (int(np.round(x2) + cols1), int(np.round(y2))), 2, q, 1)

            # Draw a line in between the two points
            # thickness = 1
            # colour blue
            cv2.line(out, (int(np.round(x1)), int(np.round(y1))), (int(np.round(x2) + cols1), int(np.round(y2))),
                     q, thickness=3)  # 画线，cv2.line()参考官方文档

    # Also return the image if you'd like a copy
    return out
