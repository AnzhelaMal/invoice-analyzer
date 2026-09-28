import pytesseract
import cv2
import numpy as np
import pandas as pd

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
# 2. FIX PERSPECTIVE
# =========================

def find_document_contour(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 30, 100)
    kernel = np.ones((5, 5), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)

    cv2.imwrite("temp_2/document_edges.png", edges)

    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    document_contour = None

    image_area = image.shape[0] * image.shape[1]

    for contour in contours:

        area = cv2.contourArea(contour)

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
    return document_contour

original = cv2.imread("data/nakladna.jpg")
document_contour = find_document_contour(original)


# 3. І ТІЛЬКИ ТЕПЕР використовуємо document_contour

def aligned_image (image, document_contour):
    debug_img = image.copy()

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
            image,
            document_contour.reshape(4, 2)
        )

        cv2.imwrite(
            "temp_2/warped_document.png",
            warped
        )
        return warped

    else:
        print("Document contour not found")
        return None

warped = aligned_image(original, document_contour)
img = warped.copy()

# =========================
# 3. REMOVE SHADOWS
# =========================

def removing_shadows(image):
# Розділяємо на 3 кольорові канали (Blue, Green, Red)
    rgb_planes = cv2.split(image)

    result_norm_planes = []

    # Обробляємо кожен канал окремо для вирівнювання тіней
    for plane in rgb_planes:
        dilated_img = cv2.dilate(plane, np.ones((7, 7), np.uint8))
        bg_img = cv2.medianBlur(dilated_img, 21)
        diff_img = 255 - cv2.absdiff(plane, bg_img)
        norm_img = cv2.normalize(
            diff_img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)

        result_norm_planes.append(norm_img)

    # 4. Об'єднуємо канали назад у кольорове зображення без тіней
    result_norm = cv2.merge(result_norm_planes)
    cv2.imwrite('temp_2/shadows_out_norm.png', result_norm)

    return result_norm

result_norm = removing_shadows(img)

# =========================
# 4. PREPROCESSING
# =========================

def noise_removal(image):
    kernel = np.ones((1, 1), np.uint8)
    image = cv2.dilate(image, kernel, iterations=1)
    kernel = np.ones((1, 1), np.uint8)
    image = cv2.erode(image, kernel, iterations=1)
    image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    image = cv2.medianBlur(image, 1)
    return (image)

def preprocessing(image):
    gray_no_shadow = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    thresh, im_bw = cv2.threshold(gray_no_shadow, 200, 230, cv2.THRESH_BINARY)
    cv2.imwrite("temp_2/bw_image2.png", im_bw)

    no_noise = noise_removal(im_bw)
    cv2.imwrite("temp_2/no_noise.png", no_noise)

    blur = cv2.GaussianBlur(gray_no_shadow, (5, 5), 0)
    cv2.imwrite("temp_2/index_blur.png", blur)

    thresh = cv2.threshold(
        blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    cv2.imwrite("temp_2/index_thresh.png", thresh)


    kernal = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    cv2.imwrite("temp_2/index_kernal.png", kernal)
    return gray_no_shadow, thresh, kernal

gray_no_shadow, thresh, kernal = preprocessing(result_norm)

# =========================
# 5. FIND TEXT BLOCKS
# =========================

def find_text_blocks(image, thresh, kernal):
    dilate = cv2.dilate(thresh, kernal, iterations=1)
    cv2.imwrite("temp_2/index_dilate.png", dilate)

    # створюємо контури
    cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]


    # now we need to sort out these images
    cnts = sorted(cnts, key=lambda x: cv2.boundingRect(x)[0])

    bbox_img = image.copy()
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
        if w > image.shape[1] * 0.8 or h > image.shape[0] * 0.5:
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

def find_table_lines(image, gray_no_shadow):
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
        (image.shape[1] // 30, 1)
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
        (1, max(20, image.shape[0] // 50))
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

    return table_mask
table_mask = find_table_lines(img, gray_no_shadow)

# =========================
# 7. CROP TABLE
# =========================

def crop_table(image, table_mask):
    # Знаходимо всі білі пікселі сітки таблиці
    points = cv2.findNonZero(table_mask)

    if points is not None:

        # Прямокутник навколо всієї знайденої сітки
        x, y, w, h = cv2.boundingRect(points)

        x1 = max(0, x)
        y1 = max(0, y)

        x2 = min(image.shape[1], x + w)
        y2 = min(image.shape[0], y + h)

        # Вирізаємо таблицю саме з ВИРІВНЯНОГО зображення
        table_roi = image[y1:y2, x1:x2]
        table_mask_roi = table_mask[y1:y2, x1:x2]

        cv2.imwrite(
            "temp_2/final_table.png",
            table_roi
        )
        return table_roi, table_mask_roi

    else:
        print("Table not found")
        return None, None

table_roi, table_mask_roi = crop_table(img, table_mask)

def detect_grid_lines(table_roi, table_mask_roi):
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

        a,b = lines.shape
        for i in range(a):
            cv2.line(table_roi, (lines[i][0], lines[i][1]), (lines[i][2], lines[i][3]), (0, 0, 255), 3, cv2.LINE_AA)
            cv2.imwrite('temp_2/houghlines5.jpg',table_roi)

    print("Vertical X:", vertical_positions)
    print("Horizontal Y:", horizontal_positions)

    return vertical_positions, horizontal_positions

vertical_positions, horizontal_positions = detect_grid_lines(
    table_roi,
    table_mask_roi
)

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

def create_grid(table_roi, vertical_positions, horizontal_positions):
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
    return x_lines, y_lines

x_lines, y_lines = create_grid(
    table_roi,
    vertical_positions,
    horizontal_positions
)

# =========================
# 7. coordinates and cross out
# =========================

def split_into_cells(table_roi, x_lines, y_lines):
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
    return cells

cells = split_into_cells(table_roi, x_lines, y_lines)


# =========================
# 8. OCR
# =========================

def recognize_text(cells):
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
    return ocr_table

ocr_table = recognize_text(cells)

# =========================
# 9. VALIDATION
# =========================

def validate_table(ocr_table):
    for row in ocr_table:
        for i in range(len(row)):

            if row[i] == "":
                row[i] = "UNKNOWN"

    return ocr_table


ocr_table = validate_table(ocr_table)


# =========================
# 9. SAVE TO EXCEL
# =========================

def save_to_excel(ocr_table):
    df = pd.DataFrame(ocr_table)

    df.to_excel(
        "results/result.xlsx",
        index=False,
        header=False
    )
save_to_excel(ocr_table)