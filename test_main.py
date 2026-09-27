import pytesseract
import cv2
import numpy as np


# =========================
# FUNCTIONS
# =========================

def noise_removal(image):
    ...
    return image


def order_points(pts):
    ...
    return rect


def four_point_transform(image, pts):
    ...
    return warped


# =========================
# 1. LOAD IMAGE
# =========================

img = cv2.imread("data/nakladna.jpg")


# =========================
# 2. FIX PERSPECTIVE
# =========================

# пошук документа
# warpPerspective
# результат -> warped


# =========================
# 3. REMOVE SHADOWS
# =========================

# твій код із rgb_planes


# =========================
# 4. PREPROCESSING
# =========================

# grayscale
# threshold
# noise removal


# =========================
# 5. FIND TABLE
# =========================

# horizontal lines
# vertical lines
# table structure
# crop table


# =========================
# 6. FIND CELLS / ROWS
# =========================

# це зробимо наступним етапом


# =========================
# 7. OCR
# =========================

# pytesseract


# =========================
# 8. VALIDATION
# =========================

# barcode / article / name ...


# =========================
# 9. SAVE TO EXCEL
# =========================