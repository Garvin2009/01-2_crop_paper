import os
from PIL import Image

def crop_images_in_folder(folder_path, crop_height):
    # 確認資料夾是否存在
    if not os.path.exists(folder_path):
        print(f"資料夾 {folder_path} 不存在。")
        return
    
    # 取得資料夾內的所有檔案
    files = os.listdir(folder_path)
    
    # 篩選出所有PNG檔案
    png_files = [f for f in files if f.lower().endswith('.png')]
    
    if not png_files:
        print("資料夾內沒有PNG檔案。")
        return

    # 開始處理每一個PNG檔案
    for png_file in png_files:
        file_path = os.path.join(folder_path, png_file)
        try:
            # 開啟圖片
            with Image.open(file_path) as img:
                width, height = img.size
                
                # 設定裁切區域：保留上半部分，去掉下方的資訊
                crop_box = (0, 0, width, height - crop_height)  # 左上角到寬度，去掉指定的高度
                cropped_img = img.crop(crop_box)
                
                # 儲存裁切後的圖片
                output_path = os.path.join(folder_path, f"{png_file}")
                cropped_img.save(output_path)
                print(f"已裁切並儲存 {png_file} 為 {output_path}")
        
        except Exception as e:
            print(f"處理 {png_file} 時發生錯誤: {e}")

# 用法：指定資料夾路徑和要裁切的高度
folder_path = "C:/Users/LAB1223/Downloads/01-2_crop_paper-main/rotated_112590007"  # 替換成你的資料夾路徑
crop_height = 300  # 輸入你想去掉的高度，這邊設置為100像素

crop_images_in_folder(folder_path, crop_height)
