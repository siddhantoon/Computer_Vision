import cv2
import numpy as np
import matplotlib.pyplot as plt


def canny_edge(image):
    img = np.copy(image)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    canny = cv2.Canny(gray, 50, 150)
    return canny


def remove_noise(image):

    img = np.copy(image)
    # Create a mask
    # to remove lanes from mask used various erode and dilate operations
    kernel = np.ones((4, 4), np.uint8)
    img_dilate1 = cv2.dilate(img, kernel, iterations=2)
    kernel2 = np.ones((6, 6), np.uint8)
    img_dilate1 = cv2.erode(img_dilate1, kernel2, iterations=1)
    img_dilate1 = cv2.erode(img_dilate1, kernel, iterations=2)
    kernel3 = np.ones((2, 2), np.uint8)
    mask = cv2.erode(img_dilate1, kernel3, iterations=1)
    # mask fro our region of region_of_interest

    # Now mask is white where noise is high and black over lanes
    # invert it to make lanes visible
    mask = cv2.bitwise_not(mask)

    return mask


def region_of_interest(image):

    roi_mask = np.zeros_like(image)
    polygons = np.array([
        [(0, 270), (50, 100), (485, 100), (640, 232)]
    ])
    cv2.fillPoly(roi_mask, polygons, 255)

    apply_mask = cv2.bitwise_and(image, roi_mask)

    return apply_mask


def display_lines(image, lines):
    # function that displayes lines on given image
    # lane_lines = np.zeros_like(image)
    lane_lines = np.copy(image)

    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        cv2.line(lane_lines, (x1, y1), (x2, y2), (0, 0, 255), 2)
    return lane_lines


def average_slope(lines):
    # print("in average slope")
    left_fit = []
    right_fit = []
    left_lane = []
    right_lane = []
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        print(parameters)
        slope = parameters[0]
        intercept = parameters[1]

        if slope < (-1/8):
            left_fit.append((slope, intercept))
        elif slope > (1/8):
            right_fit.append((slope, intercept))
    # print("before checking left fit")
    # print("left fit  ", left_fit, "right fit  ", right_fit)
        # sometimes lines aren't detected
    if left_fit:
        left_fit_average = np.average(left_fit, axis=0)
        left_lane = make_coordinates(left_fit_average)
    if right_fit:
        right_fit_average = np.average(right_fit, axis=0)
        right_lane = make_coordinates(right_fit_average)

    lanes = [left_lane, right_lane]

    return lanes


def make_coordinates(parameters):
    slope, intercept = parameters
    # print("slope", slope, "intercept", intercept)
    y1 = 100
    y2 = 290
    x1 = int((y1-intercept)/slope)
    x2 = int((y2 - intercept)/slope)
    return ([x1, y1, x2, y2])


def draw_avg_lanes(image, lanes):
    lane_lines = np.zeros_like(image)
    for lane in lanes:
        if lane:
            x1, y1, x2, y2 = lane
            cv2.line(lane_lines, (x1, y1), (x2, y2), (0, 50, 255), 8)
            cv2.line(lane_lines, (x1, y1), (x2, y2), (0, 0, 255), 2)
    return lane_lines


def lane_detection(image):
    lane_image = np.copy(image)
    canny = canny_edge(lane_image)

    noise_removal_mask = remove_noise(canny)
    roi_mask = region_of_interest(noise_removal_mask)

    final_edge = cv2.bitwise_and(roi_mask, canny)
    print("final edge")

    lines = cv2.HoughLinesP(final_edge, 2, np.pi/180, 100,
                            np.array([]), minLineLength=40, maxLineGap=40)

    if lines is not None:
        print("lines is not none")

        average_lanes = average_slope(lines)

        disp_lanes = draw_avg_lanes(lane_image, average_lanes)

        combo_image = cv2.addWeighted(lane_image, 1, disp_lanes, 0.6, 1)
    else:
        combo_image = lane_image

    # cv2.imshow('mehnat ', final_edge)
    # cv2.imshow("lanes ", display_lines(lane_image, lines))
    # cv2.imshow('average lanes', combo_image)

    # plt.imshow(lane_image)
    # cv2.waitKey(0)
    # plt.show()
    return combo_image


if __name__ == '__main__':
    # for a single image

    # image = cv2.imread('mahi_road/rd_10.jpg')
    # final = lane_detection(image)
    # cv2.imshow('original', image)
    # cv2.imshow('final', final)
    # cv2.waitKey(0)

    # for video
    cap = cv2.VideoCapture("mahi_road.mp4")
    while (cap.isOpened()):
        _, frame = cap.read()
        output = lane_detection(frame)
        cv2.imshow('original video', frame)
        cv2.imshow('output', output)
        cv2.waitKey(4)
