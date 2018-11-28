import tensorflow as tf
import os, sys
import functools
import numpy as np
import time
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import functional_ops
tfgan = tf.contrib.gan

session = tf.InteractiveSession()

# A smaller BATCH_SIZE reduces GPU memory usage, but at the cost of a slight slowdown
BATCH_SIZE = 64

# Run images through Inception.
inception_images = tf.placeholder(tf.float32, [BATCH_SIZE, 3, None, None])
activations1 = tf.placeholder(tf.float32, [None, None], name = 'activations1')
activations2 = tf.placeholder(tf.float32, [None, None], name = 'activations2')
fcd = tfgan.eval.frechet_classifier_distance_from_activations(activations1, activations2)

def inception_activations(images = inception_images, num_splits = 1):
    images = tf.transpose(images, [0, 2, 3, 1])
    size = 299
    images = tf.image.resize_bilinear(images, [size, size])
    generated_images_list = array_ops.split(images, num_or_size_splits = num_splits)
    activations = functional_ops.map_fn(
        fn = functools.partial(tfgan.eval.run_inception, output_tensor = 'pool_3:0'),
        elems = array_ops.stack(generated_images_list),
        parallel_iterations = 1,
        back_prop = False,
        swap_memory = True,
        name = 'RunClassifier')
    activations = array_ops.concat(array_ops.unstack(activations), 0)
    return activations

activations =inception_activations()

def get_inception_activations(inps):
    n_batches = inps.shape[0]//BATCH_SIZE
    act = np.zeros([n_batches * BATCH_SIZE, 2048], dtype = np.float32)
    for i in range(n_batches):
        inp = inps[i * BATCH_SIZE:(i + 1) * BATCH_SIZE] / 255. * 2 - 1
        act[i * BATCH_SIZE:(i + 1) * BATCH_SIZE] = activations.eval(feed_dict = {inception_images: inp})
    return act

def activations2distance(act1, act2):
     return fcd.eval(feed_dict = {activations1: act1, activations2: act2})
        
def get_fid(images1, images2):
    assert(type(images1) == np.ndarray)
    assert(len(images1.shape) == 4)
    assert(images1.shape[1] == 3)
    assert(np.min(images1[0]) >= 0 and np.max(images1[0]) > 10), 'Image values should be in the range [0, 255]'
    assert(type(images2) == np.ndarray)
    assert(len(images2.shape) == 4)
    assert(images2.shape[1] == 3)
    assert(np.min(images2[0]) >= 0 and np.max(images2[0]) > 10), 'Image values should be in the range [0, 255]'
    assert(images1.shape == images2.shape), 'The two numpy arrays must have the same shape'
    print('Calculating FID with %i images from each distribution' % (images1.shape[0]))
    start_time = time.time()
    act1 = get_inception_activations(images1)
    act2 = get_inception_activations(images2)
    fid = activations2distance(act1, act2)
    print('FID calculation time: %f s' % (time.time() - start_time))
    return fid




import cv2
data_path = 'face/face_real'
img_list = []
for _,_,files in os.walk(data_path):
    for file_name in files:
        if not (file_name.endswith('.jpg') or file_name.endswith('.png')):
            continue
        img_dir = os.path.join(data_path,file_name)
        img_arr = cv2.imread(img_dir)
        img_arr = cv2.resize(img_arr,(33,33))
        img_arr = img_arr.transpose(2,0,1)
        img_arr = np.array(img_arr.reshape((1,)+img_arr.shape))
        img_list.append(img_arr)
data = np.concatenate(img_list)


data_path = 'face/face_wgan-gp'
img_list = []
for _,_,files in os.walk(data_path):
    for file_name in files:
        if not (file_name.endswith('.jpg') or file_name.endswith('.png')):
            continue
        img_dir = os.path.join(data_path,file_name)
        img_arr = cv2.imread(img_dir)
#        img_arr = cv2.imresize(img_arr,28,28)
        img_arr = img_arr.transpose(2,0,1)
        img_arr = np.array(img_arr.reshape((1,)+img_arr.shape))
        img_list.append(img_arr)
data1 = np.concatenate(img_list)

