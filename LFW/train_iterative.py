import numpy as np
import argparse
parser = argparse.ArgumentParser()
import os
from cv_util.b64_visualize import show_tile
import matplotlib.pyplot as plt

def hard_tanh(img):
    img[img>1] = 1
    img[img<-1] = -1
    img = (img+1) / 2.0 * 255
    img = img.astype(np.uint8)
    return img
# Required arguments: input and output files.
parser.add_argument(
    "solver_stage01",
    help="solver proto."
)
parser.add_argument(
    "gpu_stage01",
    help="gpu id"
)
parser.add_argument(
    "iter01",
    type=int,
    help="iter for stage01"
)
# Required arguments: input and output files.
parser.add_argument(
    "solver_stage02",
    help="solver proto."
)
parser.add_argument(
    "gpu_stage02",
    help="gpu id"
)
parser.add_argument(
    "iter02",
    type=int,
    help="iter for stage02"
)
parser.add_argument(
    "--snapshot",
    default="",
    help="snapshot of model"
)

parser.add_argument(
    "--weights",
    default="",
    help="weights of model"
)
parser.add_argument(
    "--log_dir",
    default="",
    help="log dir"
)
parser.add_argument(
    "--log_name01",
    default="stage01",
    help="log name"
)
parser.add_argument(
    "--log_name02",
    default="stage02",
    help="log name"
)
parser.add_argument(
    "--K",
    type=int,
    default=5,
    help="log name"
)
args = parser.parse_args()
log_dir = args.log_dir
K = args.K

if log_dir=="":
    log_dir = os.path.dirname(args.solver_stage01)+"/"+'log'

import os
os.environ["GLOG_alsologtostderr"] = 'true'
os.environ["GLOG_log_dir"]= log_dir
import caffe
train_stage01 = caffe.TrainManager(args.solver_stage01,args.gpu_stage01,args.snapshot,args.weights,args.log_name01)
train_stage02 = caffe.TrainManager(args.solver_stage02,args.gpu_stage02,"","",args.log_name02)

train_iter01 = 0
train_iter02 = 0
iter01 = args.iter01
iter02 = args.iter02

train_stage01.net.share_with(train_stage02.net)
net = train_stage01.net
count = 0
save_iter = 200
while True:
    print "train stage 01 iter: %d to %d" %(train_iter01*iter01,(train_iter01+1)*iter01)
    net = train_stage01.Train(np.int((train_iter01+1)*iter01),net)
    train_iter01 = train_iter01+1
    #train_stage02.ShareTrainedLayersWith(net)
    #train_stage02.net.share_with(train_stage01.net)
    print "train stage 02 iter: %d to %d"%(train_iter02*iter02,(train_iter02+1)*iter02)
    net = train_stage02.Train(np.int((train_iter02+1)*iter02),net)
    train_iter02 = train_iter02+1
    #train_stage01.ShareTrainedLayersWith(net)
    #train_stage02.net.share_with(train_stage01.net)

    if count % save_iter == 0:
        net.forward()
        imgs = net.blobs['conv8_g'].data
        imgs = np.transpose(imgs,[0,2,3,1])
        all_imgs = []
        for i in range(imgs.shape[0]):
            img_tmp = hard_tanh(imgs[i])
            all_imgs.append(img_tmp)
        show_imgs = show_tile(all_imgs,144,4)[:,:,::-1]
        plt.imshow(show_imgs)
        plt.draw()
        #plt.show(block=False)
        plt.pause(0.001)
        plt.imsave('fig/exp2_epoch_%d.png'%(count / save_iter),show_imgs)
    count = count+1




