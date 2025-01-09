import cv2
import numpy as np
import pytesseract

# ---*---

def roi(img, x, x_w, y, y_h):
    return img[y:y_h, x:x_w]

def on_EVENT_LBUTTONDOWN(event, x, y, flags, param):
    global vertices
    if event == cv2.EVENT_LBUTTONDOWN:
        vertices.append([x, y])
        try:
            cv2.imshow("window", img)
        except NameError:
            pass
    return vertices

def get_xywh(img):
    global vertices
    vertices = []

    print('Press "ESC" to quit. ')
    cv2.namedWindow("window", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("window", on_EVENT_LBUTTONDOWN)
    while True:
        cv2.imshow("window", img)
        if cv2.waitKey(0)&0xFF==27:
            break
    cv2.destroyAllWindows()

    if len(vertices) != 4:
        print("vertices number not match")
        return -1

    x = min(vertices[0][0], vertices[1][0])
    x_w = max(vertices[2][0], vertices[3][0])
    y = min(vertices[1][1], vertices[2][1])
    y_h = max(vertices[0][1], vertices[3][1])

    cv2.imshow('img', img)
    cv2.imshow('roi(img)', roi(img, x, x_w, y, y_h))

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print(f'\n x={x}, x_w={x_w}, y={y}, y_h={y_h}\n')
    
    


# ---------- 以下需要修改 ----------

def get_state_1(img):    # 自己改
    return 0

def get_state_2(img):    # 自己改
    return 0

def get_state_3(img):    # 自己改
    return 0

def get_state_4(img):    # 自己改
    return 0


def get_P1(img):
    img = roi(img,   x=700, x_w=1340, y=1025, y_h=1030)
    canny = cv2.Canny(cv2.GaussianBlur(img,(3,3),0), 0, 100)
    value = canny.argmax(axis=-1)
    return np.median(value)

def get_P2(img):
    img_roi = roi(img, x=700, x_w=1340, y=1025, y_h=1025 + 1)
    
    g = cv2.split(img_roi)[1]  # 只需要绿色通道

    # 使用 NumPy 进行阈值操作
    img_th = (g > 25) & (g < 70)  # 转换为布尔矩阵

    # 矢量化计算连续非零长度
    def longest_continuous_true(row):
        diff = np.diff(np.concatenate([[0], row, [0]]))  # 检测连续块的开始和结束
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]
        return (ends - starts).max() if len(starts) > 0 else 0

    max_length = max(longest_continuous_true(row) for row in img_th)
    return 640 - max_length

# def get_P2(img):
#     img_roi = roi(img,x=700, x_w=1340, y=1025, y_h=1025+1)

#     b, g ,r =cv2.split(img_roi)    # 颜色通道分离

#     retval, img_th = cv2.threshold(g, 25, 255, cv2.THRESH_TOZERO)             # 图像阈值处理，像素点的值低于25的设置为0
#     retval, img_th = cv2.threshold(img_th, 70, 255, cv2.THRESH_TOZERO_INV)    # 图像阈值处理，像素点的值高于70的设置为0

# #     target_img = img_roi[0]
#     max_length = 0
#     for row in img_th:
#         length = 0
#         current_max = 0
#         for pixel in row:
#             if pixel > 0:  # 仅考虑非零像素点
#                 length += 1
#                 current_max = max(current_max, length)
#             else:
#                 length = 0  # 遇到非同色像素时重置计数
#         max_length = max(max_length, current_max)
    
#     return 640-max_length




# def get_P2(img):
#     img = roi(img,  x=566, x_w=1245, y=1071, y_h=1083)
#     canny = cv2.Canny(cv2.GaussianBlur(img,(3,3),0), 0, 100)
#     value = canny.argmax(axis=-1)
#     return np.median(value)

def get_P(img):
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

    img = roi(img,   x=24, x_w=314, y=19, y_h=88)
    # 转换为灰度图像

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.Canny(cv2.GaussianBlur(gray,(3,3),0), 0, 100)
    # 二值化处理
    _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)

    # 使用 Tesseract 进行识别
    digit_text = pytesseract.image_to_string(binary, config='--psm 8 digits')
    return digit_text.strip()

def get_L(img):
    img = roi(img,  x=583, x_w=1246, y=1083, y_h=1084)
    g = cv2.split(img)[2]  

    # 使用 NumPy 进行阈值操作
    img_th = (g > 0) & (g < 50)  # 转换为布尔矩阵

    # 矢量化计算连续非零长度
    def longest_continuous_true(row):
        diff = np.diff(np.concatenate([[0], row, [0]]))  # 检测连续块的开始和结束
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]
        return (ends - starts).max() if len(starts) > 0 else 0

    max_length = max(longest_continuous_true(row) for row in img_th)
    return 663-max_length




# 不够就自己添加，多了就自己删除

def get_status(img):
    return get_P2(img), get_L(img)   # 这里也要改成相应的函数名

# ---------- 以上需要修改 ----------