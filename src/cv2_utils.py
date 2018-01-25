# Utility functions for OpenCV
import cv2
import numpy as np

DEBUG = False


def img_show(img, window_name, width=None, height=None):
    """ Helper function to show an image.
    It already takes care of pausing the program until
    any key gets pressed.
    It also handles resizing the image to the specified width and/or height.
    If only one of the dimensions is provided, the image is resized
    while keeping the aspect ratio. """

    if not DEBUG:
        return

    if width or height:
        img_h = img.shape[0]
        img_w = img.shape[1]

        if width and not height:
            r = width / img_w
            height = int(img_h * r)

        if height and not width:
            r = height / img_h
            width = int(img_w * r)

        if width and height:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, width, height)

    cv2.imshow(window_name, img)
    cv2.waitKey(0)


def img_print(img, text, pos=None, col=(0, 255, 0), thickness=2):
    if pos is None:
        sz = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.65, thickness)
        pos = (5, int(sz[1]) + 10)
    cv2.putText(img, text, (pos[0], pos[1]),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, col, thickness)


def auto_canny(image, sigma=0.33):
    """A wrapper of the cv2.Canny function which tries to automatically
    calculate the minimun and maximun thresholds based on the median
    value of the image's pixels.
    The minimun and maximun values are calculated from the sigma parameter
    as 100% + sigma and 100% - sigma of the median, respectively."""

    # Compute the median of the single channel pixel intensities
    m = np.median(image)

    # Apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * m))
    upper = int(min(255, (1.0 + sigma) * m))
    edged = cv2.Canny(image, lower, upper)

    # Return the edged image
    return edged


def auto_thresh(image, thresh_min=-0.3, thresh_max=0.7):
    """A wrapper of the cv2.threshold function which tries to
    automatically calculate the lower/min and upper/max thresholds
    based on the median value of the image's pixels.
    The calculation is as follows:
        lower_threshold = (1 + thresh_min) * median
        upper_threshold = (1 + thresh_max) * median
    The default values are -0.3 and 0.7, meaning that the actual values for
    the thresholds will be 70% of the median (0.7) and 170% of the median (1.7)
    for the minimun and maximun threshold, respectively.
    @param image The image to be processed
    @param thresh_min The percentage of the median to be used summed with
     the median as the min threshold.
     @param thresh_max The percentage of the median to be used summed with
     the median as the max threshold."""

    # Compute the median of the single channel pixel intensities
    m = np.median(image)

    # Apply automatic thresholding using the computed medium
    lower_thresh = int(max(0, (1.0 + thresh_min) * m))
    upper_thresh = int(min(255, (1.0 + thresh_max) * m))

    thresh = cv2.threshold(image, lower_thresh,
                           upper_thresh, cv2.THRESH_BINARY_INV)

    return thresh


def thresh_otsu(image):
    """Simple wrapper of the cv2.threshold function
    to use the OTSU algorithm"""

    thresh = cv2.threshold(
        image, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    return thresh


def step_size(image):
    """ This function is not being used and is currently deprecated
    as it's results weren't verified.
    It's purpose is to try to measure the spread of the pixel values
    or the level of contrast regardless of space correlation.
    It tries this by simply computing the step/difference between the sorted
    pixels (thus removing and completly disregarding space correlation).
    Any derivation of it's results must be used at your own risk."""

    img = image.copy()
    img.flatten()
    img.sort()

    steps = [img[i + 1] - img[i] for i in range(len(img) - 1)]

    return steps


def filter_red_out(image):
    ''' Filter the red color out using predefined RGB range values.
    This is simply a wrapper to the function cv2_utils.filter_color_out.
    The ranges used are r=(250, 255), g=(0, 230), b=(0, 230)
    @param image The image to be processed.'''
    no_red = filter_color_out(
        image, red=(240, 255), green=(0, 230), blue=(0, 230))

    return no_red


def filter_color_out(image, red, green, blue):
    ''' Filter a color in the RGB range passed.
    @param image The image to be processed
    @param red A tuple with min and max value for the red band
    @param green A tuple with min and max value for the green band
    @param blue A tuple with min and max value for the blue band'''

    # hue=(30, 255), saturation=(150, 255), value=(50, 180)
    # hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    #  H       S         V
    # Hue, Saturation, Value
    # Color, Strength, Light
    lower_color = np.array([blue[0], green[0], red[0]])
    upper_color = np.array([blue[1], green[1], red[1]])

    mask = cv2.inRange(image, lower_color, upper_color)
    mask_not = cv2.bitwise_not(mask)
    # img_show(mask, "Mask", height=950)

    background = np.full(image.shape, 255, dtype=np.uint8)

    img_not = cv2.bitwise_and(image, image, mask=mask_not)
    img_bk = cv2.bitwise_and(background, background, mask=mask)
    # img_show(img_bk, "background", height=950)

    res = cv2.bitwise_or(img_not, img_bk)
    return res


def contour_center(c):
    """ Calculates the 'center of mass' of the contour."""

    # compute the center of the contour
    M = cv2.moments(c)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])

    return (cX, cY)


def boundingRect_contour(c=None, br=None):
    if c is not None:
        (x, y, w, h) = cv2.boundingRect(c)
    elif br is not None:
        (x, y, w, h) = br
    else:
        return None

    brc = np.array([
        [x, y],
        [x, y + h],
        [x + w, y + h],
        [x + w, y]
    ]).reshape(4, 1, 2)

    return brc
