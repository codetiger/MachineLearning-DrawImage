from PIL import Image, ImageFilter, ImageOps
import os

def buildDataSetForImages(sourcePath, targetPath):
	for f in os.listdir(sourcePath):
		if os.path.isfile(os.path.join(sourcePath, f)) and not f.startswith("."):
			print("FileName: " + f)
			image = Image.open(os.path.join(sourcePath, f))
			image = image.convert(mode='L')

			image.thumbnail((48, 48), Image.BILINEAR)
			image = image.filter(ImageFilter.FIND_EDGES)
			image = image.filter(ImageFilter.FIND_EDGES)
			image = image.filter(ImageFilter.FIND_EDGES)
			image = ImageOps.invert(image)
			image = image.filter(ImageFilter.SMOOTH)


			image = image.convert(mode='P', palette='ADAPTIVE')
			image.save(os.path.join(targetPath, f))


buildDataSetForImages("data/target/", "data/train/")