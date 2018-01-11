# answer_sheet_scanner
A Python script to scan an answer sheet outputting the alternatives marked.

# Dependencies:
 - OpenCV
 - Zbar
 
# Answer Sheet
  The script expects determined markers to be present in the answer sheet. The .xcf Gimp Project file contains a example of what the answer sheet is expected to contain.
  # Markers description
    1. It is expected to exist a barcode with the metadata for that answer sheet. It can be a QRcode, code 39 or any 2D code supported by the zbar library used.
    2. There must be four triangle markers, forming a rectangle region in which must be all the questions.
    3. Each question must have a question marker consisting of a rectangle with black contour and filled half black.
    4. All question markers must be at a determined distance from the beggining of the alternatives regions. This can be adjusted in the "define_alternatives_region" function's source code. The default is set to 75 pixels in the source code for the Zbar (or 66 pixels in the Gimp project)
    5. Every alternative must be at the same distance from each other. The default is set to 63 for the Zbar (or 62 in the Gimp project)

# Usage
  Try "python answersheet-scanner.py -h"
  
# Debugging
  The helper module cv2_utils.py has a variable called DEBUG, normally set to False. Set it to True to enable showing the various phases of the process through "cv2.imshow".
