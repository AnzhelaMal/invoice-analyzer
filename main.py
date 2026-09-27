import pytesseract
import cv2
import numpy as np


# =========================
# FUNCTIONS
# =========================


# пошук документа
def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]      # top-left
    rect[2] = pts[np.argmax(s)]      # bottom-right

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]   # top-right
    rect[3] = pts[np.argmax(diff)]   # bottom-left

    return rect


def four_point_transform(image, pts):
    rect = order_points(pts)

    tl, tr, br, bl = rect

    width_a = np.linalg.norm(br - bl)
    width_b = np.linalg.norm(tr - tl)
    max_width = int(max(width_a, width_b))

    height_a = np.linalg.norm(tr - br)
    height_b = np.linalg.norm(tl - bl)
    max_height = int(max(height_a, height_b))

    padding = 10

    dst = np.array([
        [padding, padding],
        [max_width + padding, padding],
        [max_width + padding, max_height + padding],
        [padding, max_height + padding]
    ], dtype="float32")

    matrix = cv2.getPerspectiveTransform(rect, dst)

    warped = cv2.warpPerspective(
        image,
        matrix,
        (
            max_width + padding * 2,
            max_height + padding * 2
        )
    )

    return warped


# =========================
# 1. LOAD IMAGE
# =========================

original = cv2.imread("data/nakladna.jpg")

# =========================
# 2. FIX PERSPECTIVE
# =========================

# warpPerspective

gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)

blur = cv2.GaussianBlur(gray, (5, 5), 0)

edges = cv2.Canny(blur, 30, 100)

kernel = np.ones((5, 5), np.uint8)

edges = cv2.dilate(
    edges,
    kernel,
    iterations=1
)

cv2.imwrite("temp_2/document_edges.png", edges)

contours, _ = cv2.findContours(
    edges,
    cv2.RETR_LIST,
    cv2.CHAIN_APPROX_SIMPLE
)

contours = sorted(
    contours,
    key=cv2.contourArea,
    reverse=True
)

document_contour = None

image_area = original.shape[0] * original.shape[1]

for contour in contours:

    area = cv2.contourArea(contour)

    # контур документа має займати велику частину фото
    if area < image_area * 0.3:
        continue

    perimeter = cv2.arcLength(contour, True)

    approx = cv2.approxPolyDP(
        contour,
        0.02 * perimeter,
        True
    )

    if len(approx) == 4:
        document_contour = approx
        break


# 3. І ТІЛЬКИ ТЕПЕР використовуємо document_contour
debug_img = original.copy()

if document_contour is not None:

    cv2.drawContours(
        debug_img,
        [document_contour],
        -1,
        (0, 0, 255),
        5
    )

    cv2.imwrite(
        "temp_2/detected_document.png",
        debug_img
    )

    warped = four_point_transform(
        original,
        document_contour.reshape(4, 2)
    )


    cv2.imwrite(
        "temp_2/warped_document.png",
        warped
    )

    img = warped.copy()

else:
    print("Document contour not found")

# =========================
# 3. REMOVE SHADOWS
# =========================


# 2. Розділяємо на 3 кольорові канали (Blue, Green, Red)
rgb_planes = cv2.split(img)

# result_planes = []
result_norm_planes = []

# 3. Обробляємо кожен канал окремо для вирівнювання тіней
for plane in rgb_planes:
    dilated_img = cv2.dilate(plane, np.ones((7, 7), np.uint8))
    bg_img = cv2.medianBlur(dilated_img, 21)
    diff_img = 255 - cv2.absdiff(plane, bg_img)
    norm_img = cv2.normalize(
        diff_img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)

    # result_planes.append(diff_img)
    result_norm_planes.append(norm_img)

# 4. Об'єднуємо канали назад у кольорове зображення без тіней
result_norm = cv2.merge(result_norm_planes)
cv2.imwrite('temp_2/shadows_out_norm.png', result_norm)

# =========================
# 4. PREPROCESSING
# =========================

# 5. Preprocess Images for Text OCR

# Перетворюємо оброблене зображення в градації сірого
gray_no_shadow = cv2.cvtColor(result_norm, cv2.COLOR_BGR2GRAY)

# текст став помітніший та чорно-білий
thresh, im_bw = cv2.threshold(gray_no_shadow, 200, 230, cv2.THRESH_BINARY)
cv2.imwrite("temp_2/bw_image2.png", im_bw)

# noise removal


def noise_removal(image):
    kernel = np.ones((1, 1), np.uint8)
    image = cv2.dilate(image, kernel, iterations=1)
    kernel = np.ones((1, 1), np.uint8)
    image = cv2.erode(image, kernel, iterations=1)
    image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    image = cv2.medianBlur(image, 1)
    return (image)


no_noise = noise_removal(im_bw)
cv2.imwrite("temp_2/no_noise.png", no_noise)


# 1. Blur
blur = cv2.GaussianBlur(gray_no_shadow, (5, 5), 0)
cv2.imwrite("temp_2/index_blur.png", blur)

# 2. Threshold
thresh = cv2.threshold(
    blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
cv2.imwrite("temp_2/index_thresh.png", thresh)


kernal = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
cv2.imwrite("temp_2/index_kernal.png", kernal)


# =========================
# 5. FIND TEXT BLOCKS
# =========================


# саме тут ми мржемо помітити структуру, яка вибудовується, ми все блуримо, щоб побачити column і тут ми вже можемо буде зробити bouty boxes or Contours around these columns
dilate = cv2.dilate(thresh, kernal, iterations=1)
cv2.imwrite("temp_2/index_dilate.png", dilate)

# створюємо контури
cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]


# now we need to sort out these images
cnts = sorted(cnts, key=lambda x: cv2.boundingRect(x)[0])

bbox_img = img.copy()
# draw bounding boxes only for useful contours
for c in cnts:

    # площа контуру
    area = cv2.contourArea(c)

    # координати bounding box
    x, y, w, h = cv2.boundingRect(c)

    # 1. Прибираємо дуже маленькі контури / зелені цяточки
    if area < 20:
        continue

    # 2. Прибираємо дуже маленькі bounding boxes
    if w < 5 or h < 5:
        continue

    # 3. Прибираємо величезні контури
    if w > img.shape[1] * 0.8 or h > img.shape[0] * 0.5:
        continue

    # малюємо bounding box
    cv2.rectangle(
        bbox_img,
        (x, y),
        (x + w, y + h),
        (36, 255, 12),
        2
    )

cv2.imwrite("temp_2/index_bbox.png", bbox_img)

# =========================
# 6. FIND TABLE LINES
# =========================

# Для пошуку ліній робимо ОКРЕМУ бінарну картинку
line_bw = cv2.adaptiveThreshold(
    gray_no_shadow,
    255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY_INV,
    31,
    15
)

cv2.imwrite(
    "temp_2/line_bw.png",
    line_bw
)


# -------------------------
# HORIZONTAL LINES
# -------------------------

h_kernel = cv2.getStructuringElement(
    cv2.MORPH_RECT,
    (img.shape[1] // 30, 1)
)

horizontal = cv2.erode(
    line_bw,
    h_kernel,
    iterations=1
)

horizontal = cv2.dilate(
    horizontal,
    h_kernel,
    iterations=1
)

cv2.imwrite(
    "temp_2/horizontal.png",
    horizontal
)


# -------------------------
# VERTICAL LINES
# -------------------------

v_kernel = cv2.getStructuringElement(
    cv2.MORPH_RECT,
    (1, max(20, img.shape[0] // 50))
)

vertical = cv2.erode(
    line_bw,
    v_kernel,
    iterations=1
)

vertical = cv2.dilate(
    vertical,
    v_kernel,
    iterations=1
)

cv2.imwrite(
    "temp_2/vertical.png",
    vertical
)


# -------------------------
# MERGE LINES
# -------------------------

table_mask = cv2.add(
    horizontal,
    vertical
)

cv2.imwrite(
    "temp_2/table_mask.png",
    table_mask
)

# =========================
# 7. CROP TABLE
# =========================

# Знаходимо всі білі пікселі сітки таблиці
points = cv2.findNonZero(table_mask)

if points is not None:

    # Прямокутник навколо всієї знайденої сітки
    x, y, w, h = cv2.boundingRect(points)

    x1 = max(0, x)
    y1 = max(0, y)

    x2 = min(img.shape[1], x + w)
    y2 = min(img.shape[0], y + h)

    # Вирізаємо таблицю саме з ВИРІВНЯНОГО зображення
    table_roi = img[y1:y2, x1:x2]
    table_mask_roi = table_mask[y1:y2, x1:x2]

    cv2.imwrite(
        "temp_2/final_table.png",
        table_roi
    )

else:
    print("Table not found")



# -----------------------------------------

edges = cv2.Canny(table_mask_roi,50,150,apertureSize = 3)
cv2.imwrite('temp_2/edges-50-150.jpg',edges)
minLineLength=100
# HoughLinesP знаходить окремі відрізки ліній і для кожного повертає 4 числа
lines = cv2.HoughLinesP(image=edges,rho=1,theta=np.pi/180, threshold=100,lines=np.array([]), minLineLength=minLineLength,maxLineGap=80)

vertical_positions = []
horizontal_positions = []

if lines is not None:
    for line in lines:

        x1, y1, x2, y2 = line

        # якщо лінія більше вертикальна
        if abs(x2 - x1) < abs(y2 - y1):
            x = int((x1 + x2) / 2)
            vertical_positions.append(x)

        # якщо лінія більше горизонтальна
        else:
            y = int((y1 + y2) / 2)
            horizontal_positions.append(y)

print("Vertical X:", vertical_positions)
print("Horizontal Y:", horizontal_positions)

a,b = lines.shape
for i in range(a):
    cv2.line(table_roi, (lines[i][0], lines[i][1]), (lines[i][2], lines[i][3]), (0, 0, 255), 3, cv2.LINE_AA)
    cv2.imwrite('temp_2/houghlines5.jpg',table_roi)

def merge_close_positions(values, distance=10):
    if not values:
        return []

    values = sorted(values)

    groups = [[values[0]]]

    for value in values[1:]:
        if value - groups[-1][-1] <= distance:
            groups[-1].append(value)
        else:
            groups.append([value])

    merged = []

    for group in groups:
        merged.append(int(np.mean(group)))

    return merged


x_lines = merge_close_positions(
    vertical_positions,
    distance=10
)

y_lines = merge_close_positions(
    horizontal_positions,
    distance=10
)

print("X lines:", x_lines)
print("Y lines:", y_lines)

grid_debug = table_roi.copy()

for x in x_lines:
    cv2.line(
        grid_debug,
        (x, 0),
        (x, grid_debug.shape[0]),
        (0, 0, 255),
        2
    )

for y in y_lines:
    cv2.line(
        grid_debug,
        (0, y),
        (grid_debug.shape[1], y),
        (255, 0, 0),
        2
    )

cv2.imwrite(
    "temp_2/grid_debug.png",
    grid_debug
)

# =========================
# 7. coordinates and cross out
# =========================

cells = []
pad = 2

for row in range(len(y_lines) - 1):
    row_cells = []

    for column in range(len(x_lines) - 1):
        x1 = x_lines[column]
        x2 = x_lines[column + 1]

        y1 = y_lines[row]
        y2 = y_lines[row + 1]

        cell = table_roi[
            y1 + pad:y2 - pad,
            x1 + pad:x2 - pad
        ]

        row_cells.append(cell)

        cv2.imwrite(
            f"cells/r{row}_c{column}.png",
            cell
        )

    cells.append(row_cells)



# =========================
# 7. OCR
# =========================




# =========================
# 8. VALIDATION
# =========================

# barcode / article / name ...


# =========================
# 9. SAVE TO EXCEL
# =========================
# =========================
# 7. OCR
# =========================

ocr_table = []

for row in cells:
    text_row = []

    for cell in row:
        text = pytesseract.image_to_string(
            cell,
            lang="ukr+eng",
            config="--psm 6"
        )

        text = text.strip()

        text_row.append(text)

    ocr_table.append(text_row)

# =========================
# 8. VALIDATION
# =========================

for row in ocr_table:
    for i in range(len(row)):

        # якщо комірка порожня
        if row[i] == "":
            row[i] = "UNKNOWN"


# =========================
# 9. SAVE TO EXCEL
# =========================

import pandas as pd

df = pd.DataFrame(ocr_table)

df.to_excel(
    "result.xlsx",
    index=False,
    header=False
)