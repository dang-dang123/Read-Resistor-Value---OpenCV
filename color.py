import cv2

# Đọc ảnh
image = cv2.imread("E:/Anh/Screenshot 2025-01-10 161650.png")

# Chuyển đổi sang không gian màu HSV
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Trích xuất giá trị HSV tại 1 pixel (ví dụ pixel tại (100, 100))


h, s, v = hsv_image[50, 20]
print(f"Hue: {h}, Saturation: {s}, Value: {v}")