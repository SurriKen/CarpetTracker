import os

from PIL import Image, ImageDraw

from parameters import ROOT_DIR

imp1 = 'datasets/test 16_cam 1_0s-639s/frames/04424.png'
imp2 = 'datasets/test 16_cam 2_0s-691s/frames/04424.png'

img1 = Image.open(os.path.join(ROOT_DIR, imp1)).convert('RGBA')
# img1.show()
img2 = Image.open(os.path.join(ROOT_DIR, imp2)).convert('RGBA')
# img2.show()
coords1_in = [[230, 210], [410, 700], [650, 690], [825, 530], [690, 35]]
coords1_out = [[145, 230], [355, 750], [720, 750], [955, 495], [765, 0]]
coords2_in = [[100, 0], [100, 215], [240, 285], [310, 200], [310, 0]]
coords2_out = [[50, 0], [50, 225], [240, 340], [365, 240], [364, 0]]


def draw_polygons(polygons: list, image: Image, fill=None, outline='blue') -> Image:
    xy = []
    for i in polygons:
        xy.append(i[0])
        xy.append(i[1])
    img2 = image.copy()
    print(image.size)
    w = int(img2.height * 0.01)
    draw = ImageDraw.Draw(img2)
    draw.polygon(xy, fill=fill, outline=outline, width=w)

    img3 = Image.blend(image, img2, 0.8)
    return img3


x1 = draw_polygons(polygons=coords1_in, image=img1)
x1 = draw_polygons(polygons=coords1_out, image=x1, outline='red')
x1.show()
x2 = draw_polygons(polygons=coords2_in, image=img2)
x2 = draw_polygons(polygons=coords2_out, image=x2, outline='red')
x2.show()
