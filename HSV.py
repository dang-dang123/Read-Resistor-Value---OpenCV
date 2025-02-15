import colorsys
import cv2
import numpy as np

def hsv_to_rgb(h, s, v):
    """Chuyển đổi giá trị HSV (H từ 0-179, S và V từ 0-255) sang RGB"""
    # Chuẩn hóa H từ 0-179 về 0-1, S và V từ 0-255 về 0-1
    h = h / 179.0
    s = s / 255.0
    v = v / 255.0
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return int(r * 255), int(g * 255), int(b * 255)

def display_color(h, s, v):
    """Hiển thị màu từ giá trị HSV"""
    rgb_color = hsv_to_rgb(h, s, v)
    print(f"RGB Color: {rgb_color}")

    # Tạo ảnh với màu sắc tương ứng
    color_img = np.zeros((300, 300, 3), dtype=np.uint8)
    color_img[:, :] = rgb_color[::-1]  # OpenCV sử dụng định dạng BGR

    cv2.imshow("Color Display", color_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        h = float(input("Nhập giá trị Hue (0-179): "))
        s = float(input("Nhập giá trị Saturation (0-255): "))
        v = float(input("Nhập giá trị Value (0-255): "))
        
        if not (0 <= h <= 179 and 0 <= s <= 255 and 0 <= v <= 255):
            print("Giá trị nhập không hợp lệ. Vui lòng thử lại!")
        else:
            display_color(h, s, v)
    except ValueError:
        print("Lỗi: Vui lòng nhập số hợp lệ!")
