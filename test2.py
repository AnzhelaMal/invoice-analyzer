import pytesseract
import cv2
import numpy as np

# 1. Завантажуємо кольорове зображення
img = cv2.imread('data/nakladna.jpg')

# 2. Розділяємо на 3 кольорові канали (Blue, Green, Red)
rgb_planes = cv2.split(img)

result_planes = []
result_norm_planes = []

# 3. Обробляємо кожен канал окремо для вирівнювання тіней
for plane in rgb_planes:
    dilated_img = cv2.dilate(plane, np.ones((7, 7), np.uint8))
    bg_img = cv2.medianBlur(dilated_img, 21)
    # cv2.imwrite('temp/bg_img.png', bg_img)
    diff_img = 255 - cv2.absdiff(plane, bg_img)
    norm_img = cv2.normalize(
        diff_img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)

    result_planes.append(diff_img)
    result_norm_planes.append(norm_img)

# 4. Об'єднуємо канали назад у кольорове зображення без тіней
result_norm = cv2.merge(result_norm_planes)
cv2.imwrite('temp/shadows_out_norm.png', result_norm)

# ---------------------------------------------------------
# 5. Preprocess Images for Text OCR

# Перетворюємо оброблене зображення в градації сірого
gray_no_shadow = cv2.cvtColor(result_norm, cv2.COLOR_BGR2GRAY)

# текст став помітніший та чорно-білий
thresh, im_bw = cv2.threshold(gray_no_shadow, 200, 230, cv2.THRESH_BINARY)
cv2.imwrite("temp/bw_image2.png", im_bw)

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
cv2.imwrite("temp/no_noise.png", no_noise)


# 1. Blur
blur = cv2.GaussianBlur(gray_no_shadow, (5, 5), 0)
cv2.imwrite("temp/index_blur.png", blur)

# 2. Threshold
thresh = cv2.threshold(
    blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
cv2.imwrite("temp/index_thresh.png", thresh)


kernal = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
cv2.imwrite("temp/index_kernal.png", kernal)



# саме тут ми мржемо помітити структуру, яка вибудовується, ми все блуримо, щоб побачити column і тут ми вже можемо буде зробити bouty boxes or Contours around these columns
dilate = cv2.dilate(thresh, kernal, iterations=1)
cv2.imwrite("temp/index_dilate.png", dilate)

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

cv2.imwrite("temp/index_bbox.png", img)

