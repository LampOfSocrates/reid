import cv2
import numpy as np
import os
import sys

INPUT_DIR = "Solutions"
OUTPUT_DIR = "fix_images1_outputs"


def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]   # top-left:     smallest x+y
    rect[2] = pts[np.argmax(s)]   # bottom-right: largest  x+y
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # top-right:   smallest x-y
    rect[3] = pts[np.argmax(diff)]  # bottom-left: largest  x-y
    return rect


def four_point_transform(image, pts):
    rect = order_points(pts)
    tl, tr, br, bl = rect
    w = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
    h = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))
    dst = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, M, (w, h))


def find_screen_quad_canny(gray):
    """Primary: Canny edges -> largest convex quadrilateral."""
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    edges = cv2.Canny(blurred, 20, 80)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=2)

    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    img_area = gray.shape[0] * gray.shape[1]
    for cnt in contours[:15]:
        if cv2.contourArea(cnt) < 0.05 * img_area:
            break
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            return approx.reshape(4, 2).astype("float32")
    return None


def find_screen_quad_threshold(gray):
    """Fallback: threshold bright screen region -> convex hull -> quad."""
    # Screen content is typically bright; ceiling/floor is darker or uniform
    _, bright = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
    bright = cv2.morphologyEx(bright, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    bright = cv2.morphologyEx(bright, cv2.MORPH_OPEN,  np.ones((20, 20), np.uint8))

    contours, _ = cv2.findContours(bright, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    img_area = gray.shape[0] * gray.shape[1]
    for cnt in contours[:5]:
        if cv2.contourArea(cnt) < 0.05 * img_area:
            break
        hull = cv2.convexHull(cnt)
        peri = cv2.arcLength(hull, True)
        approx = cv2.approxPolyDP(hull, 0.02 * peri, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            return approx.reshape(4, 2).astype("float32")
    return None


def find_screen_quad_hough(gray, image_shape):
    """Second fallback: Hough lines -> intersect to get 4 corners."""
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 30, 100)

    h, w = image_shape[:2]
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80,
                             minLineLength=int(min(h, w) * 0.2),
                             maxLineGap=20)
    if lines is None:
        return None

    # Separate into roughly-horizontal and roughly-vertical lines
    horiz, vert = [], []
    for x1, y1, x2, y2 in lines[:, 0]:
        angle = np.degrees(np.arctan2(abs(y2 - y1), abs(x2 - x1)))
        if angle < 30:
            horiz.append((x1, y1, x2, y2))
        elif angle > 60:
            vert.append((x1, y1, x2, y2))

    if len(horiz) < 2 or len(vert) < 2:
        return None

    def line_eq(x1, y1, x2, y2):
        # Returns (a, b, c) for ax + by = c
        a = y2 - y1
        b = x1 - x2
        c = a * x1 + b * y1
        return a, b, c

    def intersect(l1, l2):
        a1, b1, c1 = line_eq(*l1)
        a2, b2, c2 = line_eq(*l2)
        det = a1 * b2 - a2 * b1
        if abs(det) < 1e-6:
            return None
        x = (c1 * b2 - c2 * b1) / det
        y = (a1 * c2 - a2 * c1) / det
        return x, y

    # Pick top/bottom horizontal lines and left/right vertical lines
    horiz.sort(key=lambda l: (l[1] + l[3]) / 2)
    vert.sort(key=lambda l: (l[0] + l[2]) / 2)
    top_h, bot_h = horiz[0], horiz[-1]
    left_v, right_v = vert[0], vert[-1]

    corners = [
        intersect(top_h, left_v),
        intersect(top_h, right_v),
        intersect(bot_h, right_v),
        intersect(bot_h, left_v),
    ]
    if any(c is None for c in corners):
        return None

    return np.array(corners, dtype="float32")


def save_debug(image, quad, path):
    """Save an overlay showing the detected quadrilateral."""
    debug = image.copy()
    if quad is not None:
        pts = quad.astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(debug, [pts], True, (0, 255, 0), 4)
        for i, (x, y) in enumerate(quad.astype(int)):
            cv2.circle(debug, (x, y), 10, (0, 0, 255), -1)
            cv2.putText(debug, str(i), (x + 12, y + 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)
    scale = min(1.0, 1000 / max(debug.shape[:2]))
    if scale < 1.0:
        debug = cv2.resize(debug, (int(debug.shape[1] * scale), int(debug.shape[0] * scale)))
    cv2.imwrite(path, debug)


def process_image(path, output_dir):
    image = cv2.imread(path)
    if image is None:
        print(f"  [SKIP] Cannot read: {path}")
        return False

    h, w = image.shape[:2]
    scale = min(1.0, 1200 / max(h, w))
    proc = cv2.resize(image, (int(w * scale), int(h * scale))) if scale < 1.0 else image.copy()
    gray = cv2.cvtColor(proc, cv2.COLOR_BGR2GRAY)

    quad = find_screen_quad_canny(gray)
    method = "canny"
    if quad is None:
        quad = find_screen_quad_threshold(gray)
        method = "threshold"
    if quad is None:
        quad = find_screen_quad_hough(gray, proc.shape)
        method = "hough"

    stem = os.path.splitext(os.path.basename(path))[0]

    if quad is None:
        print(f"  [FAIL] {os.path.basename(path)} — no quad found")
        save_debug(proc, None, os.path.join(output_dir, f"DEBUG_FAIL_{stem}.jpg"))
        return False

    # Scale quad back to original resolution
    if scale < 1.0:
        quad = quad / scale

    warped = four_point_transform(image, quad)

    out_name = f"{stem}_fixed.jpg"
    cv2.imwrite(os.path.join(output_dir, out_name), warped)
    save_debug(image, quad, os.path.join(output_dir, f"DEBUG_{stem}.jpg"))

    print(f"  [OK/{method:9s}] {os.path.basename(path):50s} -> {warped.shape[1]}x{warped.shape[0]}")
    return True


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    images = sorted([
        os.path.join(INPUT_DIR, f)
        for f in os.listdir(INPUT_DIR)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])

    if not images:
        print(f"No images found in {INPUT_DIR}/")
        sys.exit(1)

    print(f"Processing {len(images)} images  ->  {OUTPUT_DIR}/\n")
    ok = sum(process_image(p, OUTPUT_DIR) for p in images)
    print(f"\nDone: {ok}/{len(images)} succeeded.")


if __name__ == "__main__":
    main()
