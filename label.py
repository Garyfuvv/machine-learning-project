import os
import copy
 
def write_txt(path,txt_path):
    num =len(os.listdir(path))
    file_path = txt_path
    file = open (file_path, 'w')
    c = os.listdir(path)
    for category in c:
        if category == '0':
            C = c.index(category)
            for imgs in os.listdir(os.path.join(path,category)):
                if imgs.endswith('.jpg'):
                    file.write(category+'/'+imgs+'|'+ '0'+ '\n')
        if category == '1':
            C = c.index(category)
            for imgs in os.listdir(os.path.join(path,category)):
                if imgs.endswith('.jpg'):
                    file.write(category+'/'+imgs+'|'+ '1'+ '\n')
    return
 
if __name__ == "__main__":
    write_txt('train','train.txt')
    write_txt('test', 'test.txt')
    write_txt('val', 'val.txt')
