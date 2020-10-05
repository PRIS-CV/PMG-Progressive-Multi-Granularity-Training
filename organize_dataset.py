import os

split = {}

train_count = {}
test_count  = {} 


with open('classes.txt') as fp:
   line = fp.readline()
   while line:
       line = line.strip()
       class_id , class_name = line.split(' ')
       folder_name = class_name.split('.')[0]
       train_count[folder_name] = 1
       test_count[folder_name]  = 1
       os.system('mkdir ./dataset/train/{}'.format('class_' + str(folder_name)))
       os.system('mkdir ./dataset/test/{}'.format('class_' + str(folder_name)))
       line = fp.readline()

with open('train_test_split.txt') as fp:
   line = fp.readline()
   while line:
       line = line.strip()
       image_id , image_split = line.split(' ')
       split[int(image_id)] = 'train' if int(image_split) else 'test'
       line = fp.readline()

with open('images.txt') as fp:
   line = fp.readline()
   while line:
       line = line.strip()
       image_id , image_path = line.split(' ')
       img_class = image_path.split('.')[0]
       image_split = split[int(image_id)]
       full_image_path = r'./images/{}'.format(image_path)
       if image_split == 'train':
        iter = train_count[img_class]
        os.system('cp {} ./dataset/train/class_{}/{}.jpg'.format(full_image_path , img_class , str(iter)))
        train_count[img_class] += 1
       else:
        iter = test_count[img_class]
        os.system('cp {} ./dataset/test/class_{}/{}.jpg'.format(full_image_path , img_class , str(iter)))
        test_count[img_class] += 1
       line = fp.readline()

