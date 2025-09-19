def clampbox(x1, y1, x2, y2, W, H):
    x1 = max(0, min(int(x1), W-1))
    y1 = max(0, min(int(y1), H-1))
    x2 = max(0, min(int(x2), W-1))
    y2 = max(0, min(int(y2), H-1))
    return x1, y1, x2, y2