import caffe
caffe.set_mode_gpu()
net = caffe.Net('test.prototxt','../../distilling/ResNet-50-model_t.caffemodel',caffe.TEST)
net.forward()
print net.blobs['accuracy'].data