# import the necessary packages
from imutils.perspective import four_point_transform
from imutils import contours
import imutils
import cv2
import numpy as np

DEBUG = True


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


# define the dictionary of digit segments so we can identify
# each digit on the thermostat
DIGITS_LOOKUP = {
    (1, 1, 1, 0, 1, 1, 1): 0,
    (0, 0, 1, 0, 0, 1, 0): 1,
    (1, 0, 1, 1, 1, 0, 1): 2,
    (1, 0, 1, 1, 0, 1, 1): 3,
    (0, 1, 1, 1, 0, 1, 0): 4,
    (1, 1, 0, 1, 0, 1, 1): 5,
    (1, 1, 0, 1, 1, 1, 1): 6,
    (1, 0, 1, 0, 0, 1, 0): 7,
    (1, 1, 1, 1, 1, 1, 1): 8,
    (1, 1, 1, 1, 0, 1, 1): 9
}

# load the example image
image = cv2.imread("tests/digit-test.png")

# pre-process the image by resizing it, converting it to
# graycale, blurring it, and computing an edge map
image = imutils.resize(image, height=500)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
edged = cv2.Canny(blurred, 50, 200, 255)

# find contours in the edge map, then sort them by their
# size in descending order
cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
                        cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if imutils.is_cv2() else cnts[1]
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
displayCnt = None
img = image.copy()
cv2.drawContours(img, cnts, -1, (0, 0, 255), 3)
img_show(img, "Contours", height=950)

# loop over the contours
for c in cnts:
    # approximate the contour
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)

    # if the contour has four vertices, then we have found
    # the thermostat display
    if len(approx) == 4:
        displayCnt = approx
        break

# extract the thermostat display, apply a perspective transform
# to it
displayCnt = displayCnt.reshape(4, 2)
warped = four_point_transform(gray, displayCnt)
output = four_point_transform(image, displayCnt)
warped = warped[10:-10, 10:-10]
output = output[10:-10, 10:-10]
img_show(output, "Warped", height=950)

# threshold the warped image, then apply a series of morphological
# operations to cleanup the thresholded image
thresh = cv2.threshold(warped, 0, 255,
                       cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 5))
thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
img_show(thresh, "Thresh", height=950)

# find contours in the thresholded image, then initialize the
# digit contours lists
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                        cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if imutils.is_cv2() else cnts[1]

img = output.copy()
cv2.drawContours(img, cnts, -1, (0, 0, 255), 3)
img_show(img, "Digits Contours", height=950)

digitCnts = []
widths = []
heights = []
width_dict = {}
# loop over the digit area candidates
for c in cnts:
    # compute the bounding box of the contour
    (x, y, w, h) = cv2.boundingRect(c)
    widths.append(w)
    heights.append(h)
    print("w:", w, "h:", h)
    img = output.copy()
    cv2.drawContours(img, [c], -1, (0, 255, 0), 1)
    img_print(img, "w: %s h: %s" % (w, h))
    # cv2.putText(img, "w: %s h: %s" % (w, h), (x - 10, y - 10),
    # cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
    img_show(img, "Digit", height=950)
    # if the contour is sufficiently large, it must be a digit
    if w >= 5 and (h >= 30 and h <= 40):
        digitCnts.append(c)

for w in widths:
    if w in width_dict.keys():
        width_dict[w] += 1
    else:
        width_dict[w] = 1

most_commom_width = sorted(width_dict, key=width_dict.get, reverse=True)[0]
estimated_width = np.ceil(np.mean(heights) / 1.5)
digit_width = int(np.ceil(np.mean([most_commom_width, estimated_width])))

# sort the contours from left-to-right, then initialize the
# actual digits themselves
digitCnts = contours.sort_contours(digitCnts,
                                   method="left-to-right")[0]

img = output.copy()
cv2.drawContours(img, digitCnts, -1, (0, 0, 255), 3)
img_show(img, "Digits?", height=950)

digits = []

# loop over each of the digits
for c in digitCnts:
    # extract the digit ROI
    (x, y, w, h) = cv2.boundingRect(c)
    # Fix the width of the contour to account for the digit one
    diff_w = digit_width - w
    x = x - diff_w
    w = digit_width
    roi = thresh[y:y + h, x:x + w]

    # compute the width and height of each of the 7 segments
    # we are going to examine
    (roiH, roiW) = roi.shape
    (dW, dH) = (int(roiW * 0.25), int(roiH * 0.15))
    dHC = int(roiH * 0.05)

    # define the set of 7 segments
    segments = [
        ((0, 0), (w, dH)),  # top
        ((0, 0), (dW, h // 2)),  # top-left
        ((w - dW, 0), (w, h // 2)),  # top-right
        ((0, (h // 2) - dHC), (w, (h // 2) + dHC)),  # center
        ((0, h // 2), (dW, h)),  # bottom-left
        ((w - dW, h // 2), (w, h)),  # bottom-right
        ((0, h - dH), (w, h))   # bottom
    ]
    on = [0] * len(segments)
    img = output.copy()
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 1)
    # loop over the segments
    for (i, ((xA, yA), (xB, yB))) in enumerate(segments):
        # extract the segment ROI, count the total number of
        # thresholded pixels in the segment, and then compute
        # the area of the segment
        segROI = roi[yA:yB, xA:xB]
        total = cv2.countNonZero(segROI)
        area = (xB - xA) * (yB - yA)
        cv2.rectangle(img, (xA + x, yA + y),
                      (xB + x, yB + y), (255, 0, 0), 1)
        img_show(img, "Digits Analisys", height=950)

        # if the total number of non-zero pixels is greater than
        # 50% of the area, mark the segment as "on"
        if total / float(area) > 0.5:
            on[i] = 1

    print(on)
    # lookup the digit and draw it on the image
    try:
        digit = DIGITS_LOOKUP[tuple(on)]
    except KeyError as e:
        print(e)
        digit = -1
    digits.append(digit)
    cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 1)
    cv2.putText(output, str(digit), (x + w // 2, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

# display the digits
print(u"{}{}.{} \u00b0C".format(*digits))
cv2.imshow("Input", image)
cv2.imshow("Output", output)
cv2.waitKey(0)
