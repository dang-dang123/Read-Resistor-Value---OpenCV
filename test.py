import numpy as np
import cv2

# Define color ranges in HSV for resistor bands
COLOR_RANGES = {
    "black": [(0, 0, 0), (180, 255, 50)],       
#   "brown": [(10, 100, 20), (24, 255, 220)],  
    "red": [(0, 120, 50), (10, 255, 255)],     
    "orange": [(10, 200, 100), (25, 255, 255)],
    "yellow": [(25, 150, 150), (35, 255, 255)],
    "green": [(35, 100, 50), (85, 255, 255)],  
    "blue": [(85, 100, 50), (125, 255, 255)],  
    "violet": [(125, 50, 50), (150, 255, 255)],
#     "gray": [(0, 0, 50), (180, 50, 200)],      
#     "white": [(0, 0, 200), (180, 30, 255)],    
}


COLOR_VALUES = {
    "black": 0,
    "brown": 1,
    "red": 2,
    "orange": 3,
    "yellow": 4,
    "green": 5,
    "blue": 6,
    "violet": 7,
  #  "gray": 8,
  #  "white": 9,
}


def preprocess_image(image):
 
    resized = cv2.resize(image, (800, 400)) 
    blurred = cv2.GaussianBlur(image, (5, 5), 0)  
    cv2.imshow("1",blurred);
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)  
    cv2.imshow("2",hsv);
    return hsv, image 


def merge_overlapping_bands(detected_colors):
    """
    Hợp nhất các dải màu gần nhau hoặc chồng lấn để tránh nhận diện trùng lặp.
    """
    merged_colors = []
    # Sắp xếp theo vị trí x (chiều ngang)
    detected_colors.sort(key=lambda item: item[1])  # Sort by x position

    for color, start, end in detected_colors:
        if not merged_colors or merged_colors[-1][2] < start - 10:  # Không trùng lặp
            # Nếu không có vùng chồng lấn, thêm vào danh sách mới
            merged_colors.append([color, start, end])
        else:
            # Nếu có vùng chồng lấn, hợp nhất với vùng trước đó
            merged_colors[-1][2] = max(merged_colors[-1][2], end)

    # Trả về danh sách dải màu đã hợp nhất
    return [(color, start) for color, start, _ in merged_colors]

def detect_colors(hsv_image, color_ranges, output_image):
    """
    Detect vertical color bands in the resistor image based on HSV ranges.
    Also, draw green rectangles around detected color bands.
    """
    detected_colors = []

    # Define region of interest (ROI)
    height, width, _ = hsv_image.shape
    roi_top = int(height * 0.4)
    roi_bottom = int(height * 0.8)
    roi = hsv_image[roi_top:roi_bottom, :]  # Extract the central region

    for color, (lower, upper) in color_ranges.items():
        lower_bound = np.array(lower, dtype=np.uint8)
        upper_bound = np.array(upper, dtype=np.uint8)

        # Create a mask for the current color
        mask = cv2.inRange(roi, lower_bound, upper_bound)

        # Morphological operations to reduce noise
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # Find contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h
            aspect_ratio = h / w if w != 0 else 0

            # Filter based on vertical aspect ratio and area
            if area > 150 and aspect_ratio > 0.7:  # Ensure band-like shape
                detected_colors.append((color, x, x + w))  # Save color with start and end positions

    # Hợp nhất các dải màu trùng lặp
    merged_colors = merge_overlapping_bands(detected_colors)

    # Vẽ các dải màu hợp nhất
    for color, x in merged_colors:
        x1, y1, x2, y2 = x, roi_top, x + 20, roi_bottom  # x + 20 là chiều rộng ước lượng
        cv2.rectangle(output_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Chỉ trả về danh sách các màu (không cần vị trí)
    return [color for color, _ in merged_colors]

def calculate_resistance(colors):
   
    if len(colors) < 3:
        return "Error: Not enough bands detected"

    try:
        resistance = 0  
        if len(colors) == 3:
            # 4-band resistor
            value = COLOR_VALUES[colors[0]] * 10 + COLOR_VALUES[colors[1]]
            multiplier = 10 ** COLOR_VALUES[colors[2]]
            # tolerance = colors[3]  
            resistance = value * multiplier

        elif len(colors) == 4:
            # 5-band resistor
            value = (
                COLOR_VALUES[colors[0]] * 100 +
                COLOR_VALUES[colors[1]] * 10 +
                COLOR_VALUES[colors[2]]
            )
            multiplier = 10 ** COLOR_VALUES[colors[3]]
            # tolerance = colors[4]  
            resistance = value * multiplier

        unit = " Ohm"

        if resistance >= 1_000_000:
            resistance /= 1_000_000
            unit = " MOhm"
        elif resistance >= 1_000:
            resistance /= 1_000
            unit = " kOhm"

        return f"{resistance:.1f}{unit}"
    except KeyError:
        return "Error: Invalid color detected"




def main():

    image_path = "E:/Opencvpython/bf6de6e8-d4d7-41e2-bbfd-c68a04da8762 (1).jpg"
    image = cv2.imread(image_path)

    if image is None:
        print("Error: Image not found.")
        return

    hsv_image, resized = preprocess_image(image)
    detected_colors = detect_colors(hsv_image, COLOR_RANGES, resized)
    print("Detected colors:", detected_colors)

    resistance = calculate_resistance(detected_colors)
    print("Resistance:", resistance)

    # Draw detected bands and resistance
    for i, color in enumerate(detected_colors):
        cv2.putText(resized, color, (50, 50 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.putText(resized, resistance, (400, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.imshow("Detected Colors and Resistance", resized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
