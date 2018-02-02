# cartaoresposta-scanner.py
# python cartaoresposta-scanner.py --image img_1.png

# import the necessary packages
import numpy as np
from math import pi as PI, inf
import argparse
from imutils import contours
import imutils
from imutils.perspective import four_point_transform
import cv2
import cv2_utils
# import zbar
import csv
import os
from glob import glob
from pathlib import Path
import shutil


def abort(errorMsg="Error! Aborting."):
    import sys
    sys.exit(errorMsg)


def load_image(img):
    # Load the image
    image = cv2.imread(img)

    if image is not None:
        cv2_utils.img_show(image, "Original", height=950)

    return image


def read_args():
    # Construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--outfile", required=False,
                    help="Path to the output file")
    ap.add_argument("-s", "--sucess_dir", required=False, default="sucessos",
                    help="Path to the folder to which will be moved the \
                    sucessfully processed image files.")
    ap.add_argument("-f", "--fail_dir", required=False, default="falhas",
                    help="Path to the folder to which will be moved the image\
                     files processed with fail.")
    ap.add_argument("--debug", required=False, action='store_true',
                    help="Flag to enable the debug showing of the images")
    gr = ap.add_mutually_exclusive_group(required=True)
    gr.add_argument("-i", "--image", help="Path to the input image")
    gr.add_argument("-d", "--imgdir", help="Path to the directory with the images to be processed.\
                    Will process all images found in this directory based on \
                    file extension.The extensions lookep up for are: [.png; \
                    .jpg; .jpeg; .gif; .bmp.].")
    args = vars(ap.parse_args())

    cv2_utils.DEBUG = args["debug"]

    ap = "'" if args["outfile"] is not None else ""
    print("Output File: %s%s%s" % (ap, args["outfile"], ap))
    if args["imgdir"] is not None:
        print("Images directory: '%s'" % args["imgdir"])
        print("Sucess directory: '%s'" % args["sucess_dir"])
        print("Fail directory: '%s'" % args["fail_dir"])
    else:
        print("Image: '%s'" % args["image"])

    return (args["image"], args["outfile"], args["imgdir"],
            args["sucess_dir"], args["fail_dir"])


def list_images(path):
    extensions = ["*.png", "*.jpg", "*.jpeg", "*.gif", "*.bmp"]
    files = []
    for ext in extensions:
        files.extend(glob(os.path.join(path, ext)))

    return files


def decode_qrcode(gray_img, original_img=None):
    ''' Decodes the qrcode'''
    scanner = zbar.Scanner()
    results = scanner.scan(gray_img)

    print(results)
    if len(results) == 0:
        print("No code found. Terminating")
        return None

    print("Found %d codes." % len(results))
    if original_img is not None:
        for code in results:
            print(code)
            cv2.polylines(original_img, [np.array(
                code.position)], True, (0, 0, 255), 3)

            cv2.circle(original_img, code.position[0], 5, (0, 0, 255), 3)
            cv2.circle(original_img, code.position[1], 5, (0, 255, 0), 3)
            cv2.circle(original_img, code.position[2], 5, (255, 0, 0), 3)
            cv2.circle(original_img, code.position[3], 5, (255, 0, 255), 3)

            cv2_utils.img_show(original_img, "qrcodes", height=950)

    if len(results) > 1:
        print("Warning: Found more than one QRcode, will use the first\
            with correct header")

    return results


def parse_qrcode_data(data):
    ''' Parses the code inside the QRcode.
    The code should be in the following format:
    SEDUCE;school;test;student;n_questions;n_alternatives;rows:columns

    @param data The string data to be parsed.
    @returns A dictionary with the fields read.
    All fields are integers, except "format" which is a list of integers.'''

    # school.test.student.n_questions.n_alternatives.rows-columnsX
    # 52020959.000002440.11001529449.5.5.5-1E

    data = data.decode('UTF-8')
    fields = data.split(".")

    if len(fields) == 6:
        formato = [int(i) for i in fields[5][:-1].split("-")]
        if len(formato) == 2:
            try:
                data = {
                    "school": int(fields[0]),
                    "test": int(fields[1]),
                    "student": int(fields[2]),
                    "n_questions": int(fields[3]),
                    "n_alternatives": int(fields[4])
                }

                data["format"] = formato
                return data

            except ValueError:
                print("Wrong QRcode data: " +
                      "invalid type in one of the fields!")

        else:
            print("Wrong QRcode data: " +
                  "Invalid format!")
    else:
        print("Wrong QRcode data: " +
              "Wrong number of fields! Expected 6, got %d" % len(fields))

    return None


def preprocess(image):
    ''' def preprocess(image) -> (gray, edged)
    Converts the image to grayscale, blur it slightly
     (to remove high frequency noise), then applies edge detection.
     @param image: the image to be preprocessed
     @returns A tuple with the image converted to gray as the first element
     and the resulting image of edge detection '''
    blurred = cv2.bilateralFilter(image, 5, 175, 175)
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)

    edged = cv2_utils.auto_canny(gray)
    edged_dilate = cv2.dilate(edged, np.ones((3, 3), np.uint8), iterations=1)
    edged_erode = cv2.erode(edged_dilate, np.ones(
        (3, 3), np.uint8), iterations=1)

    cv2_utils.img_show(blurred, "Blurred", height=950)
    cv2_utils.img_show(gray, "Gray", height=950)
    cv2_utils.img_show(edged, "Edged", height=950)
    cv2_utils.img_show(edged_dilate, "Edged Dilated", height=950)
    cv2_utils.img_show(edged_erode, "Edged Eroded", height=950)

    return (gray, edged_erode)


def find_markers(edged, image=None):
    ''' Finds the contours in the edge map and tries
     to identify the triangle-looking ones, that should be the markers.'''
    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if imutils.is_cv2() else cnts[1]

    img = image.copy()
    cv2.drawContours(img, cnts, -1, (0, 0, 255), 3)
    cv2_utils.img_show(img, "Contours", height=950)

    markers = []
    markers_area = []
    # Ensure that at least one contour has been found
    if len(cnts) > 0:
        # Sort the contour according to their area size in descending order
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

        # Loop over the sorted contours
        for c in cnts:
            area = cv2.contourArea(c)
            if area > 100:
                # Approximate the contour to a simpler shape
                peri = cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, 0.06 * peri, True)

                # If the approximated contour has three points,
                # then it's assumed to have found a marker.
                # convex = cv2.isContourConvex(c)
                # print(len(approx), area, convex)
                # img2 = image.copy()
                # cv2.drawContours(img2, [approx], -1, (0, 0, 255), 2)
                # cv2_utils.img_show(img2, "Contorno", height=950)
                # img3 = image.copy()
                # for pt in approx:
                #     cv2.circle(img3, tuple(pt[0]), 3, (0, 255, 0), 2)
                #     cv2_utils.img_show(img3, "contour points", height=950)

                if len(approx) == 3:
                    markers_area.append(cv2.contourArea(approx))
                    markers.append(approx)

                    # If for some reason it has found more than 4 triangles
                    # We keep only the ones with the lower contour area std
                    while len(markers) > 4:
                        mean = np.mean(markers_area)
                        diff_mean = sorted(
                            [(abs(mean - v), i)
                                for i, v in enumerate(markers_area)],
                            reverse=True)
                        del markers_area[diff_mean[0][1]]
                        del markers[diff_mean[0][1]]
        if len(markers) < 4:
            print("Couldn't find all the four markers." +
                  " Cannot continue. Aborting!")
            return None
    else:
        print(
            "Error! No contour found! Impossible to continue. Aborting!")
        return None

    if len(markers) > 0:
        img = image.copy()
        cv2.drawContours(img, markers, -1, (255, 0, 255), 2)
        cv2_utils.img_show(img, "Markers", height=950)
    else:
        print("Error! No marker found! Giving up.")
        return None

    return markers


def warpcrop(image, gray, points):
    ''' Apply a four point perspective transform to the grayscale image
     to obtain a top-down view of the paper. '''
    ansROI = four_point_transform(image, points.reshape(4, 2))[0]
    warped, m_transform = four_point_transform(gray, points.reshape(4, 2))
    cv2_utils.img_show(warped, "Warped", height=800)

    return (ansROI, warped, m_transform)


def read_student_id(thresh, markers, m_transform, img=None):
    # Offsets taken from the Gimp project
    top_offset = 2.12
    bottom_offset = 0.42

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
        (1, 1, 1, 0, 0, 1, 0): 7,
        (1, 1, 1, 1, 1, 1, 1): 8,
        (1, 1, 1, 1, 0, 1, 1): 9
    }

    bottom_markers = contours.sort_contours(
        markers, method="bottom-to-top")[0][:2]
    m_bbox = contours.sort_contours(
        bottom_markers, method="right-to-left")[1]

    heigths = [i[-1] for i in m_bbox]
    markers_height = int(np.mean(heigths))
    top = m_bbox[0][1] - (markers_height * top_offset)
    bottom = m_bbox[0][1] - (markers_height * bottom_offset)

    id_region = np.array([
        [m_bbox[1][0] + m_bbox[1][2], top],
        [m_bbox[0][0], top],
        [m_bbox[0][0], bottom],
        [m_bbox[1][0] + m_bbox[1][2], bottom],
    ], dtype='float32')
    id_region_warped = cv2.perspectiveTransform(
        id_region.reshape(1, 4, 2), m_transform)
    id_region_warped = id_region_warped.reshape(4, 2)

    # cv2.drawContours(img, id_region.reshape(4, 1, 2), -1, (0, 0, 255), 3)
    # for pt in id_region_warped:
    #     cv2.circle(img, tuple(pt), 3, (0, 0, 255), 2)
    # cv2_utils.img_show(img, "Id Region", height=950)
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 5))
    # thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    thresh = cv2.dilate(thresh, (3, 3), iterations=1)
    idRoi = thresh[int(id_region_warped[0][1]):int(id_region_warped[3][1]),
                   int(id_region_warped[0][0]):int(id_region_warped[1][0])]
    idRoi = imutils.resize(idRoi, width=1500)
    imgRoi = img[int(id_region_warped[0][1]):int(id_region_warped[3][1]),
                 int(id_region_warped[0][0]):int(id_region_warped[1][0])]
    imgRoi = imutils.resize(imgRoi, width=1500)

    cv2_utils.img_show(idRoi, "Id ROI")
    # find contours in the thresholded image, then initialize the
    # digit contours lists
    cnts = cv2.findContours(idRoi, cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if imutils.is_cv2() else cnts[1]

    img2 = imgRoi.copy()
    cv2.drawContours(img2, cnts, -1, (0, 0, 255), 3)
    # cv2_utils.img_show(img2, "Digits Contours")

    digitCnts = []
    widths = []
    heights = []
    # loop over the digit area candidates
    for c in cnts:
        # compute the bounding box of the contour
        (x, y, w, h) = cv2.boundingRect(c)
        widths.append(w)
        heights.append(h)
        # print("w:", w, "h:", h)
        # img2 = imgRoi.copy()
        # cv2.drawContours(img2, [c], -1, (0, 255, 0), 1)
        # cv2_utils.img_print(img2, "w: %s h: %s" % (w, h))
        # cv2_utils.img_show(img2, "Digit")
        # if the contour is sufficiently large, it must be a digit
        if h > 35:
            if (w > 25 and w < h) or (w > 5 and (w / h) - 0.2 < 0.1):
                digitCnts.append(c)

    if len(digitCnts) == 0:
        print("No digits found!")
        return None

    mean_width = np.mean(widths)
    normalized_widths = [i for i in widths if i >= mean_width]
    digit_width = int(np.ceil(np.mean(normalized_widths)))

    # print("widths:", widths)
    # print("heights:", heights)
    # print("normalized_widths:", normalized_widths)
    # print("digit_width:", digit_width)

    # sort the contours from left-to-right, then initialize the
    # actual digits themselves
    digitCnts = contours.sort_contours(digitCnts,
                                       method="left-to-right")[0]

    img3 = imgRoi.copy()
    cv2.drawContours(img3, digitCnts, -1, (0, 0, 255), 3)
    cv2_utils.img_show(img3, "Digits?")

    digits = []

    # loop over each of the digits
    for c in digitCnts:
        # extract the digit ROI
        (x, y, w, h) = cv2.boundingRect(c)

        # print("Bellow average?", w < mean_width)
        if w < mean_width:
            # Fix the width of the contour to account for the digit one
            diff_w = digit_width - w
            x = x - diff_w
            w = digit_width

        roi = idRoi[y:y + h, x:x + w]

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
        # loop over the segments
        for (i, ((xA, yA), (xB, yB))) in enumerate(segments):
            # extract the segment ROI, count the total number of
            # thresholded pixels in the segment, and then compute
            # the area of the segment
            segROI = roi[yA:yB, xA:xB]
            total = cv2.countNonZero(segROI)
            area = (xB - xA) * (yB - yA)

            # Printing the segments
            img4 = imgRoi.copy()
            cv2.rectangle(img4, (x, y), (x + w, y + h), (0, 255, 0), 1)
            cv2.rectangle(img4, (xA + x, yA + y),
                          (xB + x, yB + y), (255, 0, 0), 1)
            cv2_utils.img_print(img4, "Black Px: %.2f%%" %
                                ((total / area) * 100), col=(0, 0, 0))
            cv2_utils.img_show(img4, "Digits Analisys")

            # if the total number of non-zero pixels is greater than
            # 70% of the area, mark the segment as "on"
            if total / area > 0.6:
                on[i] = 1

        # lookup the digit and draw it on the image
        try:
            digit = DIGITS_LOOKUP[tuple(on)]
        except KeyError as e:
            print(e)
            digit = -1
        digits.append(digit)
        cv2.rectangle(imgRoi, (x, y), (x + w, y + h), (0, 255, 0), 1)
        cv2.putText(imgRoi, str(digit), (x + w // 2, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

    # display the digits
    # print(digits)
    cv2_utils.img_show(imgRoi, "Output")

    student_id = 0
    for i, v in enumerate(digits):
        student_id = student_id + v * (10 ** (len(digits) - 1 - i))

    return student_id


def sort_questionMarks(questionMarks, questions_format):
    ''' Organizes/Sorts the question marks found according to
    the format passed '''
    rows = questions_format[0]

    questionMarks = contours.sort_contours(
        questionMarks, method="left-to-right")[0]
    qm = []
    for i in np.arange(0, len(questionMarks), rows):
        qm_col = contours.sort_contours(
            questionMarks[i:i + rows], method="top-to-bottom")[0]

        qm.extend(qm_col)

    return qm


def define_each_alts_region(alts_regions, n_alternativas, total_alt_width):
    ''' Calculates the respective individual alternative's region.'''
    # print("define_each_alts_region")
    all_alternatives = []
    alt_width = total_alt_width / n_alternativas
    for alts in alts_regions:
        startColumn = alts[0][0][0]
        rowUp = alts[0][0][1]
        rowDown = alts[1][0][1]

        alternatives = []
        for i in range(n_alternativas):
            leftColumn = startColumn + (i * alt_width)
            alt = [[[int(leftColumn), int(rowUp)]],
                   [[int(leftColumn), int(rowDown)]],
                   [[int(leftColumn + alt_width), int(rowDown)]],
                   [[int(leftColumn + alt_width), int(rowUp)]]]
            alt = np.asarray(alt)
            alternatives.append(alt)

        all_alternatives.append(alternatives)
    return all_alternatives


def define_alternatives_region(questionMarks, marker_height,
                               n_alternativas, img=None):
    ''' Calculates the alternative's regions based on
    offset and width constants. '''

    # As measured in Gimp:
    #   marker height = 43px
    #   offset = 63px = ceil(marker_height * 1.46)
    #   alt width = 62px = ceil(marker_height * 1.44)

    # After being printed and scanned the values change to 1.85 and 1.35,
    # respectively.
    # The offset is the distance in pixels from the rightmost side of the
    # question marker to the start/leftmost side of the alternative's region.
    offset = int(np.ceil(marker_height * 1.85))
    # The width of one alternative's region
    # Same as the offset but with a plus 1.
    alt_width = int(np.ceil(marker_height * 1.35))

    alts_region = []
    for c in questionMarks:
        alts = []
        for i in range(len(c)):
            alt = int(np.floor(i / 2)) * n_alternativas
            alts.append(
                [[c[i][0][0] + offset + (alt * alt_width),
                  c[i][0][1]]])

        alts = np.asarray(alts)
        alts_region.append(alts)

    for i, r in enumerate(alts_region):
        cv2.drawContours(img, [r], -1, (255, 0, 0), 5)
        c = cv2_utils.contour_center(r)
        cv2.putText(img, str(i), c, cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (0, 0, 255), 2)
    cv2_utils.img_show(img, "Alternatives regions", height=950)

    all_alternatives = define_each_alts_region(
        alts_region, n_alternativas, alt_width * n_alternativas)
    return all_alternatives


def find_questions(
        n_alternativas, n_questoes, questions_format, thresh, img=None):
    ''' Finds the questions using a serious of measures to
     ensure the identification of the question markers '''

    thresh_cnts = cv2.findContours(
        thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    thresh_cnts = thresh_cnts[0] if imutils.is_cv2() else thresh_cnts[1]

    img1 = img.copy()
    cv2.drawContours(img1, thresh_cnts, -1, (0, 255, 255), 2)
    cv2_utils.img_show(img1, "AnsROI Contours", height=800)

    questionMarks = []
    squares = []
    pentagons = []
    hexagons = []
    shape_similar = []
    smaller_area_min_circle = []
    questionMarks_heights = []

    img2 = img.copy()
    # Loop over the contours
    for c in thresh_cnts:
        # Compute the area of the contour to filter out very small noisy areas.
        area = cv2.contourArea(c)

        # Calculates the perimeter of the contour
        peri = cv2.arcLength(c, True)
        # Approximates to a polygon, using a very small epsilon,
        # to keep most of the original contour.
        approx = cv2.approxPolyDP(c, 0.03 * peri, True)

        # img3 = img.copy()
        # cv2.drawContours(img3, [approx], -1, (0, 0, 255), 3)
        # cv2_utils.img_show(img3, "Question marks loop", height=950)

        # Checks the area of the contour for very small areas,
        # as it's likely to be just noise.
        # print("Area:", area)
        if area > 10:
            # Looks for the question markers
            # accepts contours with 4, 5 and 6 vertices
            if abs(len(approx) - 5) <= 1:
                if len(approx) == 4:
                    squares.append(approx)
                elif len(approx) == 5:
                    pentagons.append(approx)
                elif len(approx) == 6:
                    hexagons.append(approx)

                br = cv2.boundingRect(c)
                brc = cv2_utils.boundingRect_contour(br=br)
                shape_diff = cv2.matchShapes(approx, brc, 1, 0)
                # print("shape_diff:", shape_diff)
                if shape_diff < 0.3:
                    shape_similar.append(brc)

                    (x, y, w, h) = br
                    questionMarks_heights.append(h)
                    circle_center, min_radius = cv2.minEnclosingCircle(approx)
                    min_circle_area = PI * (min_radius ** 2)
                    br_area = w * h

                    if min_circle_area > br_area:
                        smaller_area_min_circle.append(brc)
                        seg_div = 0.55
                        upper_offset_y = int(h / 5.5)
                        upper_offset_x = int(w / 5)

                        upper_start_y = y + upper_offset_y
                        upper_end_y = (y + int(h * seg_div))
                        upper_start_x = x + upper_offset_x
                        upper_end_x = x + w - upper_offset_x

                        lower_start_y = (y + int(h * seg_div))
                        lower_end_y = y + h
                        lower_start_x = x
                        lower_end_x = x + w

                        upperMarkRoi = thresh[
                            upper_start_y:upper_end_y,
                            upper_start_x:upper_end_x
                        ]
                        lowerMarkRoi = thresh[
                            lower_start_y:lower_end_y,
                            lower_start_x:lower_end_x
                        ]
                        segment_area = w * (h * seg_div)

                        upperNonZero = cv2.countNonZero(upperMarkRoi)
                        lowerNonZero = cv2.countNonZero(lowerMarkRoi)

                        pct_upperBlack = upperNonZero / segment_area
                        pct_lowerBlack = lowerNonZero / segment_area

                        lower_threshold = 0.7
                        upper_threshold = 0.3

                        # print("pct_upperBlack:", pct_upperBlack)
                        # print("pct_lowerBlack:", pct_lowerBlack)
                        passed = (pct_upperBlack < upper_threshold and
                                  pct_lowerBlack > lower_threshold)
                        # print("Passed?", passed)

                        cv2.rectangle(img2, (upper_start_x, upper_start_y),
                                      (upper_end_x, upper_end_y),
                                      (0, 255 * passed, 255 * (not passed)), 1)
                        cv2.rectangle(img2, (lower_start_x, lower_start_y),
                                      (lower_end_x, lower_end_y),
                                      (0, 255 * passed, 255 * (not passed)), 2)
                        cv2_utils.img_show(img2, "marker segments", height=950)

                        if (pct_upperBlack < upper_threshold and
                                pct_lowerBlack > lower_threshold):
                            questionMarks.append(brc)

    img2 = img.copy()
    cv2.drawContours(img2, questionMarks, -1, (255, 0, 0), 2)
    cv2_utils.img_show(img2, "Question marks", height=950)

    img3 = img.copy()
    # print(len(squares))
    cv2.drawContours(img3, squares, -1, (0, 0, 255), 2)
    cv2_utils.img_show(img3, "squares", height=950)
    img3 = img.copy()
    # print(len(pentagons))
    cv2.drawContours(img3, pentagons, -1, (255, 0, 255), 2)
    cv2_utils.img_show(img3, "pentagons", height=950)
    img3 = img.copy()
    # print(len(hexagons))
    cv2.drawContours(img3, hexagons, -1, (255, 0, 255), 2)
    cv2_utils.img_show(img3, "hexagons", height=950)

    img3 = img.copy()
    # print(len(shape_similar))
    cv2.drawContours(img3, shape_similar, -1, (0, 0, 255), 2)
    cv2_utils.img_show(img3, "shape_similar", height=950)

    img3 = img.copy()
    # print(len(smaller_area_min_circle))
    cv2.drawContours(img3, smaller_area_min_circle, -1, (0, 0, 255), 2)
    cv2_utils.img_show(img3, "smaller_area_min_circle", height=950)

    if len(questionMarks) != n_questoes and len(questionMarks) > 0:
        print("WARNING! Number of markers found is different from what is " +
              "stated on the metadata! Markers: %d != Questions: %d"
              % (len(questionMarks), n_questoes))
    elif len(questionMarks) == 0:
        print("No question marks found. Impossible to continue.")
        return None

    questionMarks = sort_questionMarks(questionMarks, questions_format)

    all_alternatives = define_alternatives_region(
        questionMarks, np.mean(questionMarks_heights),
        n_alternativas, img.copy())

    color = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    color = (color * (len(all_alternatives) // len(color))) + \
        color[:len(all_alternatives) % len(color)]
    for question, c in zip(all_alternatives, color):
        cv2.drawContours(img, question, -1, c, 3)
    cv2_utils.img_show(img, "Individual alternatives", height=950)

    return all_alternatives


def check_answers(all_alternatives, n_alternativas, thresh):
    ''' Counts the number of black pixels in each alternative area
    and assigns the one with the most as the marked one. '''

    answers = []
    for n, question in enumerate(all_alternatives):
        maxNonZero = 1
        minNonZero = 100
        ans = 0
        for i, alt in enumerate(question):
            # construct a mask that reveals only the current
            # region of the question
            mask = np.zeros(thresh.shape, dtype="uint8")
            cv2.drawContours(mask, [alt], -1, 255, -1)

            # apply the mask to the thresholded image, then
            # count the number of non-zero pixels in the
            # alternative area
            mask = cv2.bitwise_and(thresh, thresh, mask=mask)
            nonZero = cv2.countNonZero(mask)
            # print("Q: %d; A: %d; Px: %d" % (n, i, nonZero))

            if nonZero > minNonZero:
                if nonZero >= maxNonZero:
                    if abs(1 - (nonZero / maxNonZero)) < 0.5:
                        ans = 9
                        maxNonZero = inf
                    else:
                        ans = i + 1
                        maxNonZero = nonZero
        answers.append(ans)
        # print("Questão %d: %f" % (n, maxNonZero))
    return answers


def write_to_file(student_id, answers, outfile):
    ''' Saves the answers identified to the file.
    If no file has been passed then just prints to the console '''

    header = ["Aluno"]
    header.extend(["Questao_" + str(i) for i in range(1, len(answers) + 1)])
    print_flag = False
    if outfile is not None:
        writeHeader = False
        if not os.path.exists(outfile):
            writeHeader = True

        try:
            with open(outfile, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile, delimiter=';')
                if writeHeader:
                    writer.writerow(header)

                writer.writerow([student_id] + answers)
        except Exception as e:
            print("\nError wirting to file:", e)
            print("Falling back to printing to console.\n")
            print_flag = True
    else:
        print_flag = True

    if print_flag:
        print(header)
        print([student_id] + answers)


def process_image(image_file, outfile):
    ''' Process a image scanned of a answer sheet '''

    # Loads the image
    print("\n\nWill process:", image_file)
    image = load_image(image_file)
    if image is None:
        print("Image '%s' couldn't be loaded." % image_file)
        return False
    # print(image.shape)
    # image = imutils.resize(image, height=1800)
    # print(image.shape)

    # Filter the red color out
    image = cv2_utils.filter_red_out(image)
    cv2_utils.img_show(image, "No red", height=950)

    gray, edged = preprocess(image)

    # results = decode_qrcode(gray)
    # if results is None:
    #     # return False
    #     print("That's ok though")
    #     qrcode_data = None
    # else:
    #     for result in results:
    #         qrcode_data = parse_qrcode_data(result.data)

    #         if qrcode_data is not None:
    #             break

    # if qrcode_data is None:
    #     print("No valid QRcode found." +
    #           " Will use the default values of 84 questions, each with 5" +
    #           " alternatives, organized in 28 rows and 3 columns.")
    #     qrcode_data = {
    #         "student": 1,
    #         "n_questions": 84,
    #         "n_alternatives": 5,
    #         "format": [28, 3]
    #     }
    #     # return False

    metadata = {
        "n_questions": 81,
        "n_alternatives": 5,
        "format": [27, 3]
    }

    markers = find_markers(edged, image)
    if markers is None:
        return False

    markers_center = np.array([cv2_utils.contour_center(c) for c in markers])

    ansROI, warped, m_transform = warpcrop(image, gray, markers_center)

    thresh_val, thresh_img = cv2_utils.auto_thresh(warped)
    cv2_utils.img_show(thresh_img, "auto thresh", height=800)

    # Reads the student id
    student_id = read_student_id(
        thresh_img, markers, m_transform, ansROI.copy())

    # Resizes the cropped ROI
    thresh_img = imutils.resize(thresh_img, height=1000)
    ansROI = imutils.resize(ansROI, height=1000)

    all_alternatives = find_questions(
        metadata["n_alternatives"], metadata["n_questions"],
        metadata["format"], thresh_img.copy(), ansROI.copy())

    if all_alternatives is None:
        return False

    answers = check_answers(
        all_alternatives, metadata["n_alternatives"], thresh_img.copy())

    write_to_file(student_id, answers, outfile)
    print("Done!")
    return True


def main():
    # Reads the image passed through args
    image, outfile, imgdir, sucess_dir, fail_dir = read_args()

    if image is not None:
        process_image(image, outfile)
    else:
        Path(sucess_dir).mkdir(parents=True, exist_ok=True)
        Path(fail_dir).mkdir(parents=True, exist_ok=True)

        files = list_images(imgdir)
        if len(files) > 0:
            print("Images found:", len(files))
        else:
            print("No images found on the directory '%s'. Can't continue."
                  % imgdir)
            return

        for img in files:
            sucess = process_image(img, outfile)
            cv2.destroyAllWindows()

            try:
                if sucess is True:
                    shutil.move(img, sucess_dir)
                else:
                    shutil.move(img, fail_dir)
            except shutil.Error as e:
                print("Error while moving the original image file:", e,
                      "\nWill continue without moving.")


if __name__ == '__main__':
    main()
