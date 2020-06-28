import random
from PIL import Image
import os

all_dir = "/Users/garyfu/courses/ml/project/all"
train_dir = "/Users/garyfu/courses/ml/project/train"
test_dir = "/Users/garyfu/courses/ml/project/test"
val_dir = "/Users/garyfu/courses/ml/project/validation"


list = []
for i in range(100):
	list.append(i)

random.shuffle(list)

for i in range(60):
	j = list[i] + 1  
	img0 = Image.open(os.path.join(all_dir,'0',str(j)+'.jpg'))
	img0.save(os.path.join(train_dir,'0',str(j)+'.jpg'))
	img1 = Image.open(os.path.join(all_dir,'1',str(j)+'.jpg'))
	img1.save(os.path.join(train_dir,'1',str(j)+'.jpg'))


for i in range(60,80):
	j = list[i] + 1 
	img0 = Image.open(os.path.join(all_dir,'0',str(j)+'.jpg'))
	img0.save(os.path.join(val_dir,'0',str(j)+'.jpg'))
	img1 = Image.open(os.path.join(all_dir,'1',str(j)+'.jpg'))
	img1.save(os.path.join(val_dir,'1',str(j)+'.jpg'))


for i in range(80,100):
	j = list[i] + 1
	img0 = Image.open(os.path.join(all_dir,'0',str(j)+'.jpg'))
	img0.save(os.path.join(test_dir,'0',str(j)+'.jpg'))
	img1 = Image.open(os.path.join(all_dir,'1',str(j)+'.jpg'))
	img1.save(os.path.join(test_dir,'1',str(j)+'.jpg'))
