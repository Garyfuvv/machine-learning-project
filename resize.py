
from PIL import Image
import os.path
import glob


def convertjpg(jpgfile, outdir, width=256, height=256):
    img = Image.open(jpgfile)
    try:
        new_img = img.resize((width, height), Image.BILINEAR)
        new_img.save(os.path.join(outdir, os.path.basename(jpgfile)))
    except Exception as e:
        print(e)


for jpgfile in glob.glob("/Users/garyfu/courses/ml/project/positive/*.jpg"):  # 读取文件
    convertjpg(jpgfile, "/Users/garyfu/courses/ml/project/all/1")  # 保存文件位置

for jpgfile in glob.glob("/Users/garyfu/courses/ml/project/negative/*.jpg"):  # 读取文件
    convertjpg(jpgfile, "/Users/garyfu/courses/ml/project/all/0")  # 保存
