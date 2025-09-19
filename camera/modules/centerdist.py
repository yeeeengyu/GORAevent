def centerdist(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    acx, acy = (ax1+ax2)/2, (ay1+ay2)/2
    bcx, bcy = (bx1+bx2)/2, (by1+by2)/2
    return ((acx-bcx)**2 + (acy-bcy)**2) ** 0.5