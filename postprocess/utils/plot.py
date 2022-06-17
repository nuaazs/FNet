import ants

def plot_ct(img,slices,overlay=None,title=None):
    if overlay!=None:
        img_over = overlay - overlay.min()
    else:
        img_over = None
    img2 = img-img.min()
    img2.plot(axis=1,slices=slices,title=title,overlay=img_over,overlay_alpha=0.5)