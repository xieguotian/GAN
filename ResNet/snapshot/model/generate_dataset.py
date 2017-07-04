import caffe
import os
import numpy as np
import matplotlib.pyplot as plt
def hard_tanh(img):
    #img[img>1] = 1
    #img[img<-1] = -1
    #img = (img+1) / 2.0 * 255
    img = img + np.array([104.007,116.669,122.679]).reshape([1,1,3])
    img[img>255] = 255
    img[img<0] = 0
    img = img.astype(np.uint8)
    return img

caffe.set_mode_gpu()
caffe.set_device(6)
net = caffe.Net('stage02.prototxt','stage02_iter_45000_2vs5.caffemodel',caffe.TRAIN)

batch_size = 32
num_image = 10000
num_iter = num_image / batch_size + 1

count = 0
print num_iter

with open('image_list2.txt','w') as fid:
    for i in range(num_iter):
        print i
        net.forward()
        img = net.blobs['re_conv'].data
        label = net.blobs['label'].data
        print 'accuracy: %f ' % net.blobs['accuracy'].data
        img = np.transpose(img,[0,2,3,1])
        for j in range(label.shape[0]):
            count+=1
            save_folder = 'dataset2/%05d'%(label[j])
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)
            save_name = save_folder + '/%09d.png'%(count)
            print >>fid,'%s\t%d'%(save_name,label[j])

            image_tmp = hard_tanh(img[j])[:,:,::-1]
            plt.imsave(save_name,image_tmp)

