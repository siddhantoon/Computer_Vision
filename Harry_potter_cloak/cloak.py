import cv2
import numpy as np

cloak_color_low = np.array([10, 0, 20])
cloak_color_high = np.array([50, 255, 255])


def roi(image):

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    # cv2.imshow("img", image)

    mask = cv2.inRange(image, cloak_color_low, cloak_color_high)

    return mask


def invisible(img, bg):
    # img = cv2.imread('')
    mask = roi(img)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)

    negative_mask = cv2.bitwise_not(mask)
    # cv2.imshosw("neg", negative_mask)
    # cv2.imshow("mask", mask)
    cloak = cv2.bitwise_and(bg, bg, mask=mask)
    non_cloak = cv2.bitwise_and(img, img, mask=negative_mask)
    output = non_cloak + cloak
    return output


if __name__ == '__main__':

    # image = cv2.imread("imgs/opencv_frame_4.jpg")
    # img = np.copy(image)
    bg = cv2.imread("background.jpg")
    background = np.copy(bg)

    cap = cv2.VideoCapture(0)
    print("Press Escape to Quit (x)")

    while True:
        ret, frame = cap.read()

        harry = invisible(frame, background)
        cv2.imshow("Cloak", harry)
        k = cv2.waitKey(1)

        if k % 256 == 27:
            print("Escape Hit, closing...")
            break

    # harry = invisible(img, background)
    # cv2.imshow("original ", image)
    # cv2.imshow("invisible", harry)
    # cv2.waitKey(0)
