# pixel_ops.py
import cv2, os, argparse
import numpy as np

def ensure_dir(p): 
    os.makedirs(p, exist_ok=True)

def image_stats(img):
    h, w = img.shape[:2]
    mean_bgr = img.reshape(-1,3).mean(axis=0)
    return {"w": w, "h": h, "mean_bgr": mean_bgr.tolist()}

def rect_mask(h, w, x, y, rw, rh):
    m = np.zeros((h, w), dtype=np.uint8)
    x2, y2 = min(x+rw, w), min(y+rh, h)
    m[y:y2, x:x2] = 255
    return m

def poly_mask(h, w, pts):
    m = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(m, [np.int32(pts)], 255)
    return m

def color_range_mask(img_bgr, lower_hsv, upper_hsv):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    return cv2.inRange(hsv, np.array(lower_hsv), np.array(upper_hsv))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/sample.png")
    ap.add_argument("--outdir", default="outputs")
    ap.add_argument("--rect", type=int, nargs=4, metavar=("x","y","w","h"),
                    help="Square ROI (x y w h)")
    ap.add_argument("--poly", type=int, nargs="+",
                    help="Polygon ROI (x1 y1 x2 y2 ...)")
    ap.add_argument("--hsv", type=int, nargs=6, metavar=("h1","s1","v1","h2","s2","v2"),
                    help="HSV ROI")
    args = ap.parse_args()

    img = cv2.imread(args.input, cv2.IMREAD_COLOR)
    assert img is not None, f"이미지 로드 실패: {args.input}"
    h, w = img.shape[:2]
    ensure_dir(args.outdir)

    stats = image_stats(img)
    print("[INFO] Image:", stats)

    mask = np.zeros((h,w), dtype=np.uint8)
    if args.rect:
        x,y,rw,rh = args.rect
        mask = cv2.bitwise_or(mask, rect_mask(h,w,x,y,rw,rh))
    if args.poly and len(args.poly) >= 6 and len(args.poly)%2==0:
        pts = list(zip(args.poly[0::2], args.poly[1::2]))
        mask = cv2.bitwise_or(mask, poly_mask(h,w,pts))

    if args.hsv:
        h1,s1,v1,h2,s2,v2 = args.hsv
        m_color = color_range_mask(img, (h1,s1,v1), (h2,s2,v2))
        mask = cv2.bitwise_or(mask, m_color)

    extracted = cv2.bitwise_and(img, img, mask=mask)
    cv2.imwrite(os.path.join(args.outdir, "extracted.png"), extracted)

    red_boost = img.copy()
    red_boost[...,2] = np.clip(red_boost[...,2] * 1.3, 0, 255).astype(np.uint8)
    boosted = img.copy()
    boosted[mask>0] = red_boost[mask>0]
    cv2.imwrite(os.path.join(args.outdir, "boosted.png"), boosted)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thr_val, bin_img = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
    bin_masked = np.where(mask>0, bin_img, 0).astype(np.uint8)
    cv2.imwrite(os.path.join(args.outdir, "binary_masked.png"), bin_masked)

    print("[DONE] outputs saved to:", args.outdir)

if __name__ == "__main__":
    main()