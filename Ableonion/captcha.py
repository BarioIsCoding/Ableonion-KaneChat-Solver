import cv2
import numpy as np
import sys
import base64

def extend_canvas(img, ext_size):
    h, w = img.shape[:2]
    ni = np.full((h + 2 * ext_size, w + 2 * ext_size, 3), 255, dtype=np.uint8)
    ni[ext_size:ext_size + h, ext_size:ext_size + w] = img
    return ni, ext_size

def compute_hand_angle(line, cx, cy):
    x1, y1, x2, y2 = line
    # Determine which point is farther from the center (tip of the hand)
    dist1 = np.hypot(x1 - cx, y1 - cy)
    dist2 = np.hypot(x2 - cx, y2 - cy)
    if dist1 > dist2:
        x_end, y_end = x1, y1
    else:
        x_end, y_end = x2, y2
    # Adjust angle to have 12 o'clock at 0 degrees and increase clockwise
    angle = np.degrees(np.arctan2(cy - y_end, x_end - cx))
    angle = (360 - ((angle + 360) % 360) + 90) % 360
    return angle

def adjust_base64_padding(b64_string):
    """Adjust base64 string padding to ensure it's a multiple of 4."""
    return b64_string + '=' * (-len(b64_string) % 4)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <base64_image_string>")
        sys.exit(1)
    base64_image_string = sys.argv[1]

    # Adjust the base64 padding
    base64_image_string = adjust_base64_padding(base64_image_string)

    # Decode the base64 image string
    try:
        # If your base64 string uses URL-safe encoding, use urlsafe_b64decode
        nparr = np.frombuffer(base64.b64decode(base64_image_string), np.uint8)
        img1 = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    except Exception as e:
        print("Error decoding base64 image string:", e)
        sys.exit(1)

    if img1 is None:
        print("Error: Unable to load the image. Please ensure the base64 string is valid.")
        sys.exit(1)

    # First script's processing
    blurred = cv2.GaussianBlur(img1, (5, 5), 0)
    enhanced = cv2.createCLAHE(2.0, (8, 8)).apply(blurred)
    thresh = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=2)
    edges = cv2.Canny(morph, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, minLineLength=30, maxLineGap=10)

    # Circle detection to find the clock face
    height, width = img1.shape[:2]
    ext_size = 100
    out = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    ext_out, ext = extend_canvas(out, ext_size)
    ext_gray = cv2.cvtColor(ext_out, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(ext_gray, (5, 5), 0)
    enhanced = cv2.createCLAHE(2.0, (8, 8)).apply(blurred)
    edges_c = cv2.Canny(enhanced, 50, 150)
    circles = cv2.HoughCircles(edges_c, cv2.HOUGH_GRADIENT, 1.2, 50, param1=100, param2=30, minRadius=50, maxRadius=0)

    if circles is not None:
        circles = np.uint16(np.around(circles))
        c = circles[0][0]
        cx, cy, r = c
        # Adjust cx, cy due to extended canvas
        cx -= ext
        cy -= ext
    else:
        # Default values if no circle is detected
        cx, cy, r = width // 2, height // 2, min(width, height) // 2

    # If lines are detected, process them
    if lines is not None:
        ld = []
        for l in lines:
            x1, y1, x2, y2 = l[0]
            length = np.hypot(x2 - x1, y2 - y1)
            ld.append((x1, y1, x2, y2, length))
        ld.sort(key=lambda x: x[4], reverse=True)
        if len(ld) >= 2:
            # Minute hand
            angle_m = compute_hand_angle(ld[0][:4], cx, cy)
            # Hour hand
            angle_h = compute_hand_angle(ld[1][:4], cx, cy)
        elif len(ld) == 1:
            # Only minute hand detected
            angle_m = compute_hand_angle(ld[0][:4], cx, cy)
            angle_h = None
        else:
            angle_m = angle_h = None
    else:
        # No lines detected
        angle_m = angle_h = None

    # Estimate clock time based on detected lines
    if angle_m is not None:
        # Minute calculation
        minute = int(round(angle_m / 6)) % 60  # Each minute is 6 degrees

        if angle_h is not None:
            # Hour calculation using floor division
            hour = int(angle_h // 30) % 12
            if hour == 0:
                hour = 12  # Adjust 0 to 12 for display purposes
        else:
            hour = 12  # Default to 12 if hour hand not detected

        # Output the estimated time
        print(f"{hour:02d}:{minute:02d}")
    else:
        # If time could not be estimated, display a message
        print("Time could not be estimated.")
