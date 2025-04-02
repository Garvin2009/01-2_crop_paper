from PIL import Image, ImageDraw
import cv2
import numpy as np
import os
import json

def read_json(file, unicode_num):
    with open(file) as f:
        p = json.load(f)
        unicode_list = [''] * unicode_num
        for i in range(unicode_num):
            unicode_list[i] = 'U+' + p['CP950'][i]['UNICODE'][2:6]  # ex: 0x1234 --> U+1234
        return unicode_list

def scale_adjustment(word_img, img_name):
    """調整文字大小、重心
    
    Keyword arguments:
        word_img -- 文字圖片
    """
    word_img = np.array(word_img)
    # 增加更大的邊框，確保大型文字有足夠空間
    padding = 100  # 增加邊框大小
    word_img_copy = cv2.copyMakeBorder(word_img, padding, padding, padding, padding, cv2.BORDER_CONSTANT, value=(255, 255, 255))

    # 二值化處理
    binary_word_img = cv2.cvtColor(word_img_copy, cv2.COLOR_BGR2GRAY) if len(word_img_copy.shape) == 3 else word_img_copy
    binary_word_img = cv2.threshold(binary_word_img, 127, 255, cv2.THRESH_BINARY_INV)[1]

    # 取得文字 Bounding Box
    topLeftX, topLeftY, word_w, word_h = cv2.boundingRect(binary_word_img)
    max_length = max(word_w, word_h)

    # 計算質心
    cX, cY = topLeftX + word_w // 2, topLeftY + word_h // 2  # 幾何中心

    # 標註 bounding box 和質心
    annotated_img = cv2.cvtColor(word_img_copy, cv2.COLOR_GRAY2BGR) if len(word_img_copy.shape) == 2 else word_img_copy
    cv2.rectangle(annotated_img, (topLeftX, topLeftY), (topLeftX + word_w, topLeftY + word_h), (255, 168, 0), 4)
    cv2.circle(annotated_img, (cX, cY), 15, (0, 0, 255), -1)

    # 保存標註的圖片
    annotated_img_path = os.path.join('annotated_images', f'{img_name}_annotated.png')
    os.makedirs('annotated_images', exist_ok=True)
    cv2.imwrite(annotated_img_path, annotated_img)
    
    # 動態調整裁剪大小，確保能完整包含文字
    # 使用字元實際大小的1.5倍作為裁剪範圍，確保文字完整性
    crop_size = int(max(word_w, word_h) * 1.5)
    crop_size = max(crop_size, 240)  # 設定最小裁剪大小
    
    h, w = word_img_copy.shape
    left_x = max(0, cX - crop_size // 2)
    right_x = min(w, cX + crop_size // 2)
    top_y = max(0, cY - crop_size // 2)
    bot_y = min(h, cY + crop_size // 2)

    # 確保裁剪區域是正方形
    width = right_x - left_x
    height = bot_y - top_y
    if width > height:
        diff = width - height
        top_y = max(0, top_y - diff // 2)
        bot_y = min(h, bot_y + diff // 2)
    elif height > width:
        diff = height - width
        left_x = max(0, left_x - diff // 2)
        right_x = min(w, right_x + diff // 2)

    final_word_img = word_img_copy[top_y:bot_y, left_x:right_x]
    # 根據需要調整輸出大小，可能需要更大的尺寸
    output_size = 400  # 增加輸出圖像大小
    return cv2.resize(final_word_img, (output_size, output_size), interpolation=cv2.INTER_AREA)


def crop_boxes(image_folder, start_page, end_page, min_box_size, padding, json_path, unicode_num):
    # 讀取圖片
    unicode_list = read_json(json_path, unicode_num)
    k = (start_page - 1) * 100
    print(k)
    for page in range(start_page, end_page + 1):
        # 構建檔案名稱
        image_file = f"{page}.png"
        print(page)
        # 圖片路徑
        image_path = os.path.join(image_folder, image_file)

        # 讀取圖片
        image = Image.open(image_path)
        img_np = cv2.imread(image_path, cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)

        # 使用二值化處理，使方框更容易被檢測
        _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
        
        # 增加形態學操作來去除噪點
        kernel = np.ones((3, 3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        # 排除右下角的QR碼區域
        h, w = binary.shape
        qr_size = int(min(h, w) * 0.12)  # 假設QR碼大約佔圖片的12%
        binary[-qr_size:, -qr_size:] = 0  # 將右下角區域設為黑色
        # 使用輪廓檢測方框
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 對輪廓進行處理，將 y 值相差小於 10 的視為同一行
        contours = sorted(contours, key=lambda x: (cv2.boundingRect(x)[1] // 120, cv2.boundingRect(x)[0]))

        # 輪廓過濾增加更多條件
        valid_contours = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            
            # 排除右下角QR碼
            if x + w > img_np.shape[1] - qr_size and y + h > img_np.shape[0] - qr_size:
                continue
                
            # 面積過濾
            area = cv2.contourArea(contour)
            if area < min_box_size * min_box_size * 0.5:
                continue
                
            # 長寬比過濾 (排除過度細長的區域)
            aspect_ratio = float(w) / h
            if aspect_ratio < 0.5 or aspect_ratio > 2.0:
                continue
                
            # 矩形度檢查 (實際面積與理論矩形面積的比值)
            rect_area = w * h
            extent = float(area) / rect_area
            if extent < 0.6:  # 如果實際面積小於理論面積的60%，可能不是一個好的文字框
                continue
                
            valid_contours.append(contour)

        # 使用過濾後的輪廓
        contours = valid_contours

        # 確保目錄存在
        output_directory = 'crop'
        os.makedirs(output_directory, exist_ok=True)

        # 繪製藍色的邊框並裁切方框
        draw = ImageDraw.Draw(image)

        for i, contour in enumerate(contours):
            x, y, w, h = cv2.boundingRect(contour)
             # 排除右下角的QR碼區域
            if x + w > img_np.shape[1] - qr_size and y + h > img_np.shape[0] - qr_size:
                continue

            # 內縮方框
            x += padding
            y += padding
            w -= 2 * padding
            h -= 2 * padding

            # 略過小於閾值的方框
            if w >= min_box_size and h >= min_box_size:
                cropped_image = Image.fromarray(cv2.cvtColor(img_np[y:y + h, x:x + w], cv2.COLOR_BGR2RGB))
                cropped_image = np.array(cropped_image)
                cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
                median_filtered = cv2.medianBlur(cropped_image, 3)
                kernel = np.ones((2, 2), np.uint8)
                processed_image = cv2.morphologyEx(median_filtered, cv2.MORPH_OPEN, kernel)
                connectivity, labels, stats, centroids = cv2.connectedComponentsWithStats(processed_image, connectivity=8)
                for j in range(1, connectivity):
                    area = stats[j, cv2.CC_STAT_AREA]
                    if area < min_area_threshold:
                        processed_image[labels == j] = 0

                cropped_image = scale_adjustment(processed_image, unicode_list[k])
                cv2.imwrite(os.path.join(output_directory, f'{unicode_list[k]}.png'), cropped_image)
                k += 1
                cv2.rectangle(img_np, (x, y), (x + w, y + h), (255, 0, 0), 2)

                if k == unicode_num:
                    break

        bound_output_directory = 'rec_bound'
        os.makedirs(bound_output_directory, exist_ok=True)
        cv2.imwrite(os.path.join(bound_output_directory, f'{page}.png'), img_np)


if __name__ == "__main__":
    image_folder = "/home/cyantus/repo/01-2_crop_paper/1125900**" #輸入你的rotated資料夾路徑
    start_page = int(input("Enter start page: "))  # 起始頁數
    end_page = int(input("Enter end page: "))      # 結束頁數
    min_box_size = 250 # 設定閾值，只保留寬和高都大於等於這個值的方框
    min_area_threshold = 10
    padding = 15  # 內縮的像素數量
    json_path = "CP950.json"  # 請替換為你的 JSON 檔案路徑
    unicode_num = 5652 #請替換成製作稿紙時的文字量

    crop_boxes(image_folder, start_page, end_page, min_box_size, padding, json_path, unicode_num)