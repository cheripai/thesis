import os
import PIL
import sys
from PIL import Image

BASE_WIDTH = 400

if __name__ == "__main__":
    root_dir = sys.argv[1]
    target_dir = sys.argv[2]

    paths = []
    for root, dirs, files in os.walk(root_dir):
        dirs.sort()
        files.sort()
        for i, name in enumerate(files):
            paths.append(os.path.join(root, name))

    for i, path in enumerate(paths):
        img = Image.open(path)
        wpercent = (BASE_WIDTH / float(img.size[0]))
        height = int((float(img.size[1]) * float(wpercent)))
        img = img.resize((BASE_WIDTH, height), PIL.Image.ANTIALIAS)

        ext = os.path.splitext(path)[1]
        name = str(i) + ext.lower()
        img.save(os.path.join(target_dir, name))
