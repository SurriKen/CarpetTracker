from PIL import Image

img = Image.open('datasets/Train_0_0s-10s/frames/00000.png')
img.show()
print(img.size)
im = img.crop((25, 35, 120, 170))
im.show()