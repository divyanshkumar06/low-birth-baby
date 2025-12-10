import cv2
import numpy as np
import os
from imutils import contours, grab_contours

def midpoint(ptA, ptB):
    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

# --- 1. Load Image (Robust Check) ---
filename = 'baby_test.jpg'
if not os.path.exists(filename):
    if os.path.exists('baby_test.jpg.jpg'):
        filename = 'baby_test.jpg.jpg'
        print(f"⚠️ Note: Found file as '{filename}'. Loading it...")
    else:
        print(f"❌ Error: Could not find '{filename}'. Please make sure the photo is in the same folder.")
        exit()

print(f"Loading {filename}...")
image = cv2.imread(filename)

if image is None:
    print(f"❌ Error: Could not load image. File might be corrupted.")
    exit()

# Resize for consistent processing
h, w = image.shape[:2]
new_h = 800
new_w = int(w * (800/h))
image = cv2.resize(image, (new_w, new_h))

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (7, 7), 0)

# --- 2. Edge Detection & Segmentation (IMPROVED) ---
edged = cv2.Canny(gray, 30, 100)
edged = cv2.dilate(edged, None, iterations=2)
edged = cv2.erode(edged, None, iterations=1)

# Find Contours
cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = grab_contours(cnts)

print(f"🔍 Found {len(cnts)} objects/contours in the image.")

if len(cnts) == 0:
    print("❌ No objects detected. Try a photo with better lighting.")
    exit()

# Sort contours from left to right (Card -> Baby parts)
(cnts, _) = contours.sort_contours(cnts)

pixelsPerMetric = None
count = 0
total_weight_grams = 0  # Variable to store sum of all parts

# --- 3. Processing Loop ---
print("Processing objects...")
for c in cnts:
    # Filter noise
    if cv2.contourArea(c) < 2000:
        continue

    count += 1
    # Compute bounding box
    box = cv2.minAreaRect(c)
    (x, y), (w, h), angle = box
    
    # === STAGE 1: CALIBRATION ===
    if pixelsPerMetric is None:
        pixelsPerMetric = w / 8.56 
        print(f"-> [Calibration] Card Found! Ratio: {pixelsPerMetric:.2f} pixels/cm")
        
        # Visualize Card (Blue)
        box_points = cv2.boxPoints(box).astype("int")
        cv2.drawContours(image, [box_points], -1, (255, 0, 0), 2)
        cv2.putText(image, "Ref Card (8.56cm)", (int(x), int(y - 10)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        continue

    # === STAGE 2: MEASUREMENT ===
    area_pixels = cv2.contourArea(c)
    perimeter_pixels = cv2.arcLength(c, True) 
    
    # Convert to real-world units
    area_cm2 = area_pixels / (pixelsPerMetric ** 2) 
    perimeter_cm = perimeter_pixels / pixelsPerMetric
    
    # === STAGE 3: WEIGHT PREDICTION MODEL ===
    ALPHA = 0.29   
    BETA = 12.5    
    GAMMA = 120    
    
    volume_proxy = area_cm2 ** 1.5
    estimated_weight_grams = (ALPHA * volume_proxy) + (BETA * perimeter_cm) + GAMMA
    
    # Add to Total Weight
    total_weight_grams += estimated_weight_grams
    
    print(f"-> [Part Detected] Area: {area_cm2:.2f} cm², Part Weight: {estimated_weight_grams:.2f} g")

    # === VISUALIZATION ===
    # Draw Green Box around Baby Part
    box_points = cv2.boxPoints(box).astype("int")
    cv2.drawContours(image, [box_points], -1, (0, 255, 0), 2)
    
    # Display Part Weight (Small text)
    cv2.putText(image, f"{estimated_weight_grams:.0f}g", (int(x), int(y)), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

# --- 4. Display Final Total Weight ---
if total_weight_grams > 0:
    print(f"✅ Total Estimated Weight: {total_weight_grams:.2f} g")
    
    # Display Total Weight prominently at the top-left
    text = f"TOTAL WEIGHT: {total_weight_grams:.0f} g"
    cv2.rectangle(image, (10, 10), (450, 60), (0, 0, 0), -1) # Black background box
    cv2.putText(image, text, (20, 45), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3) # Yellow text

if count < 2:
    print("⚠️ Warning: Only 1 object found (Card). Baby could not be detected.")

# Show final result
cv2.imshow("Engine B: Weight Estimator", image)
print("Press any key on the image window to close...")
cv2.waitKey(0)
cv2.destroyAllWindows()