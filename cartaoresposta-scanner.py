# cartaoresposta-scanner.py
# python cartaoresposta-scanner.py --image img_1.png

# import the necessary packages
import numpy as np
import argparse
from math import floor
from imutils import contours
import imutils
from imutils.perspective import four_point_transform
import cv2
import cv2_utils
import zbar
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
    gr = ap.add_mutually_exclusive_group(required=True)
    gr.add_argument("-i", "--image", help="Path to the input image")
    gr.add_argument("-d", "--imgdir", help="Path to the directory with the images to be processed.\
                    Will process all images found in this directory based on \
                    file extension.The extensions lookep up for are: [.png; \
                    .jpg; .jpeg; .gif; .bmp.].")
    args = vars(ap.parse_args())

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

    if len(results) == 0:
        print("No QRcode found. Terminating")
        return None

    print("Found %d qrcodes." % len(results))
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

    # SEDUCE;school;test;student;n_questions;n_alternatives;rows:columns
    # SEDUCE;52020959;000002440;11001529449;5;5;5:1

    data = data.decode('UTF-8')
    fields = data.split(";")

    if fields[0] == "SEDUCE":
        if len(fields) == 7:
            formato = [int(i) for i in fields[6].split(":")]
            if len(formato) == 2:
                try:
                    data = {
                        "school": int(fields[1]),
                        "test": int(fields[2]),
                        "student": int(fields[3]),
                        "n_questions": int(fields[4]),
                        "n_alternatives": int(fields[5])
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
                  "Wrong number of fields! Expected 7, got %d" % len(fields))
    else:
        print("Wrong QRcode data: Incorrect header!")

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

    cv2_utils.img_show(blurred, "Blurred", height=950)
    cv2_utils.img_show(gray, "Gray", height=950)
    cv2_utils.img_show(edged, "Edged", height=950)

    return (gray, edged)


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
            # Approximate the contour to a simpler shape
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.04 * peri, True)

            # If the approximated contour has three points,
            # then it's assumed to have found a marker.
            if len(approx) == 3:
                markers_area.append(cv2.contourArea(approx))
                markers.append(approx)

                # If for some reason it has found more than 4 triangles
                # We keep only the ones with the lower contour area std
                while len(markers) > 4:
                    mean = np.mean(markers_area)
                    diff_mean = sorted(
                        [(abs(mean - v), i)
                            for i, v in enumerate(markers_area)], reverse=True)
                    del markers_area[diff_mean[0][1]]
                    del markers[diff_mean[0][1]]
    else:
        print(
            "Erro! Nenhum contorno encontrado! Impossível continuar. Abortar")
        return None

    if len(markers) > 0:
        img = image.copy()
        cv2.drawContours(img, markers, -1, (255, 0, 255), 3)
        cv2_utils.img_show(img, "Markers", height=950)
    else:
        print("Erro! Nenhum marcador encontrado! Desistindo.")
        return None

    return markers


def warpcrop(image, gray, points):
    ''' Apply a four point perspective transform to the grayscale image
     to obtain a top-down view of the paper. '''
    ansROI = four_point_transform(image, points.reshape(4, 2))
    warped = four_point_transform(gray, points.reshape(4, 2))
    cv2_utils.img_show(warped, "Warped", height=800)

    return (ansROI, warped)


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


def define_alternatives_region(questionMarks, n_alternativas, img=None):
    ''' Calculates the alternative's regions based on
    offset and width constants. '''

    # The distance in pixels from the rightmost side of the
    # question marker to the start/leftmost side of the alternative's region
    # The offset can be calculated using the gimp project of the answer-card
    # and using the ruler tool to measure the distance and add 9.
    offset = 75
    # The width of one alternative's region
    # Same as the offset but with a plus 1.
    alt_width = 63

    alts_region = []

    for c in questionMarks:
        alts = []
        for i in range(len(c)):
            alt = floor(i / 2) * n_alternativas
            alts.append(
                [[c[i][0][0] + offset + (alt * alt_width),
                  c[i][0][1]]])

        alts = np.asarray(alts)
        alts_region.append(alts)

    marker_width = questionMarks[0][-1][0][0] - questionMarks[0][0][0][0]
    total_alt_width = (
        alts_region[0][-1][0][0] -
        alts_region[0][0][0][0] - marker_width
    )

    for i, r in enumerate(alts_region):
        cv2.drawContours(
            img, [r], 0, (255, 0, 0), 5)
        c = cv2_utils.contour_center(r)
        cv2.putText(img, str(i), c, cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (0, 0, 255), 2)
    cv2_utils.img_show(img, "Regiões Alternativas", height=950)

    all_alternatives = define_each_alts_region(
        alts_region, n_alternativas, total_alt_width)
    return all_alternatives


def find_questions(thresh_cnts, n_alternativas, n_questoes, questions_format,
                   thresh, img=None):
    ''' Finds the questions using a serious of measures to
     ensure the identification of the question markers '''
    questionMarks = []

    # Loop over the contours
    for c in thresh_cnts:
        # Compute the area of the contour to filter out very small noisy areas.
        area = cv2.contourArea(c)

        # Calculates the perimeter of the contour
        peri = cv2.arcLength(c, True)
        # Approximates to a polygon, using a very small epsilon,
        # to keep most of the original contour.
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        # Checks the area of the contour for very small areas,
        # as it's likely to be just noise.
        if area > 100:
            # Looks for the question markers
            if len(approx) == 4:
                # construct a mask that reveals only the current
                # "mark" for the question
                mask = np.zeros(thresh.shape, dtype="uint8")
                cv2.drawContours(mask, [c], -1, 255, -1)

                # apply the mask to the thresholded image, then
                # count the number of non-zero pixels in the
                # contour area to see if it has the mark pattern
                mask = cv2.bitwise_and(thresh, thresh, mask=mask)
                total = cv2.countNonZero(mask)

                ratio_area_half = abs((area / total) - 1.5)
                if (ratio_area_half <= 0.1):
                    questionMarks.append(approx)

    if len(questionMarks) != n_questoes and len(questionMarks) > 0:
        print("WARNING! Number of markers found is different from what is " +
              "stated on the QRcode! Markers: %d != Questions: %d"
              % (len(questionMarks), n_questoes))
    elif len(questionMarks) == 0:
        print("No question marks found. Impossible to continue.")
        return None

    questionMarks = sort_questionMarks(questionMarks, questions_format)

    all_alternatives = define_alternatives_region(
        questionMarks, n_alternativas, img.copy())

    color = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    color = (color * (len(all_alternatives) // len(color))) + \
        color[:len(all_alternatives) % len(color)]
    for question, c in zip(all_alternatives, color):
        cv2.drawContours(img, question, -1, c, 3)
    cv2_utils.img_show(img, "Alternativas indviduais", height=950)

    return all_alternatives


def check_answers(all_alternatives, n_alternativas, thresh):
    ''' Counts the number of black pixels in each alternative area
    and assigns the one with the most as the marked one. '''

    answers = []
    for question in all_alternatives:
        nonzero = 0
        ans = -1
        for i, alt in enumerate(question):
            # construct a mask that reveals only the current
            # region of the question
            mask = np.zeros(thresh.shape, dtype="uint8")
            cv2.drawContours(mask, [alt], -1, 255, -1)

            # apply the mask to the thresholded image, then
            # count the number of non-zero pixels in the
            # alternative area
            mask = cv2.bitwise_and(thresh, thresh, mask=mask)
            total = cv2.countNonZero(mask)

            if total > nonzero:
                ans = i
                nonzero = total
        answers.append(ans)
    return answers


def write_to_file(school_id, test_id, student_id, answers, outfile):
    ''' Saves the answers identified to the file.
    If no file has been passed then just prints to the console '''

    header = ["Escola", "Prova", "Aluno", "Questao", "Alternativa"]
    print_flag = False
    if outfile is not None:
        writeHeader = False
        if not os.path.exists(outfile):
            writeHeader = True

        try:
            with open(outfile, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile, delimiter=',')
                if writeHeader:
                    writer.writerow(header)
                for q, r in enumerate(answers):
                    writer.writerow([school_id, test_id, student_id, q, r])
        except Exception as e:
            print("\nError wirting to file:", e)
            print("Falling back to printing to console.\n")
            print_flag = True
    else:
        print_flag = True

    if print_flag:
        print(header)
        for q, r in enumerate(answers):
            print([school_id, test_id, student_id, q, r])


def process_image(image_file, outfile):
    ''' Process a image scanned of a answer sheet '''

    # Loads the image
    print("\n\nWill process:", image_file)
    image = load_image(image_file)

    # Filter the red color out
    image = cv2_utils.filter_red_out(image)
    cv2_utils.img_show(image, "No red", height=950)

    gray, edged = preprocess(image)

    results = decode_qrcode(gray)
    if results is None:
        return False

    for result in results:
        qrcode_data = parse_qrcode_data(result.data)

        if qrcode_data is not None:
            break
    if qrcode_data is None:
        print("No valid QRcode found. Aborting.")
        return False

    markers = find_markers(edged, image)
    if markers is None:
        return False

    markers_center = np.array([cv2_utils.contour_center(c) for c in markers])

    ansROI, warped = warpcrop(image, gray, markers_center)

    thresh_val, thresh_img = cv2_utils.auto_thresh(warped)
    cv2_utils.img_show(thresh_img, "auto thresh", height=800)

    thresh_cnts = cv2.findContours(
        thresh_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    thresh_cnts = thresh_cnts[0] if imutils.is_cv2() else thresh_cnts[1]

    img = ansROI.copy()
    cv2.drawContours(img, thresh_cnts, -1, (0, 255, 255), 2)
    cv2_utils.img_show(img, "AnsROI Contours", height=800)

    all_alternatives = find_questions(
        thresh_cnts, qrcode_data["n_alternatives"], qrcode_data["n_questions"],
        qrcode_data["format"], thresh_img.copy(), ansROI.copy())
    if all_alternatives is None:
        return False

    answers = check_answers(
        all_alternatives, qrcode_data["n_alternatives"], thresh_img.copy())

    write_to_file(qrcode_data["school"], qrcode_data["test"],
                  qrcode_data["student"], answers, outfile)
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
