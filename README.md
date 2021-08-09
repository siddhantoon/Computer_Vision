# OpenCV
Image Processing using OpenCV with Python3
This is the lane detection project on an Indian street with traffic.
Algorithm is as follows:
Canny Edge=> custom noise removal mask using dilation and erosion => region of interest for lane detection => Hough line transformation=> averaging left and right lanes neglecting horizontal lines => display them on original image.
