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

    dst = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]
    ], dtype="float32")

    matrix = cv2.getPerspectiveTransform(rect, dst)

    warped = cv2.warpPerspective(
        image,
        matrix,
        (max_width, max_height)
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

    # додаємо запас навколо знайденої таблиці
    x, y, w, h = cv2.boundingRect(document_contour)

    padding = 10

    x = max(0, x - padding)
    y = max(0, y - padding)

    w = min(
        original.shape[1] - x,
        w + 2 * padding
    )

    h = min(
        original.shape[0] - y,
        h + 2 * padding
    )

    table = original[y:y+h, x:x+w]

    cv2.imwrite(
        "temp_2/table_with_padding.png",
        table
    )

else:
    print("Document contour not found")

# =========================
# 3. REMOVE SHADOWS
# =========================

img = cv2.imread("temp_2/warped_document.png")

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
        img,
        (x, y),
        (x + w, y + h),
        (36, 255, 12),
        2
    )

cv2.imwrite("temp_2/index_bbox.png", img)

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

# table_with_padding вже є таблицею,
# тому просто зберігаємо її як final_table
table_roi = img.copy()

cv2.imwrite(
    "temp_2/final_table.png",
    table_roi
)

# -----------------------------------------

edges = cv2.Canny(table_mask,50,150,apertureSize = 3)
cv2.imwrite('temp_2/edges-50-150.jpg',edges)
minLineLength=100
# HoughLinesP знаходить окремі відрізки ліній і для кожного повертає 4 числа
lines = cv2.HoughLinesP(image=edges,rho=1,theta=np.pi/180, threshold=100,lines=np.array([]), minLineLength=minLineLength,maxLineGap=80)



a,b = lines.shape
for i in range(a):
    cv2.line(table_roi, (lines[i][0], lines[i][1]), (lines[i][2], lines[i][3]), (0, 0, 255), 3, cv2.LINE_AA)
    cv2.imwrite('temp_2/houghlines5.jpg',table_roi)



# # =========================
# # 6. FIND CELLS / ROWS
# # =========================

# # shape[0] → 900  → висота
# # shape[1] → 1200 → ширина
# # shape[2] → 3    → кількість каналів кольору   і ми беремо 1, бо саме тут нас цікавить ширина картинки

# h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (img.shape[1] // 30, 1))
# # v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, img.shape[0] // 50))

# # erode — це ерозія.
# # Вона прибирає все, що не схоже на довгу горизонтальну лінію.

# # dilate — це розширення.
# # Після erode лінії можуть стати тоншими або частково пошкодитися.
# # dilate відновлює їх товщину/довжину.

# # erode + dilate  це морфологічна операція, яка фактично каже:
# # «Залиши мені структури, які схожі на довгі горизонтальні лінії».

# horizontal = cv2.erode(thresh, h_kernel, iterations=1)
# horizontal = cv2.dilate(horizontal, h_kernel, iterations=1)

# # vertical = cv2.erode(thresh, v_kernel, iterations=1)
# # vertical = cv2.dilate(vertical, v_kernel, iterations=1)

# cv2.imwrite("temp_2/horizontal.png", horizontal)
# # cv2.imwrite("temp_2/vertical.png", vertical)

# # =========================
# # VERTICAL LINES
# # =========================

# # Спочатку з'єднуємо маленькі розриви
# vertical_connect_kernel = cv2.getStructuringElement(
#     cv2.MORPH_RECT,
#     (1, 5)
# )

# vertical_source = cv2.dilate(
#     thresh,
#     cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3)),
#     iterations=1
# )

# v_kernel = cv2.getStructuringElement(
#     cv2.MORPH_RECT,
#     (1, img.shape[0] // 50)
# )

# vertical = cv2.erode(
#     vertical_source,
#     v_kernel,
#     iterations=1
# )

# vertical = cv2.dilate(
#     vertical,
#     v_kernel,
#     iterations=1
# )

# cv2.imwrite("temp_2/vertical_source.png", vertical_source)
# cv2.imwrite("temp_2/vertical.png", vertical)

# cv2.imwrite("temp_2/horizontal.png", horizontal)
# cv2.imwrite("temp_2/vertical.png", vertical)
# #  Merge Lines and Find the Table Region With Contours
# table_mask = cv2.add(horizontal, vertical)
# # Бо cv.findContours() повертає два значення. (contours, hierarchy)
# # contours — це всі знайдені контури, а друге значення: hierarchy
# # — це інформація про те, як контури пов’язані між собою.
# # Наприклад:
# # який контур знаходиться всередині іншого;
# # який є сусіднім;
# # який є батьківським або дочірнім.

# cv2.imwrite(
#     "temp_2/table_mask.png",
#     table_mask
# )

# # Але в моєму коді ця інформація не потрібна. тому прочерк contours, _
# contours, _ = cv2.findContours(
#     table_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# # cv.RETR_EXTERNAL - шукай тільки зовнішні контури.
# # cv.CHAIN_APPROX_SIMPLE -  означає, що OpenCV не буде зберігати кожну точку контуру.

# # Цей рядок шукає найбільший контур.
# table_cnt = max(contours, key=cv2.contourArea)
# # Координати таблиці
# x, y, w, h = cv2.boundingRect(table_cnt)
# # Вирізаємо таблицю
# table_roi = img[
#     y:y + h,
#     x:x + w
# ]

# cv2.imwrite(
#     "temp_2/final_table.png",
#     table_roi
# )
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
