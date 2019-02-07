import tensorflow as tf
import skimage.transform
from skimage.io import imsave, imread
import os

##### Omkar Thawakar #########
##### Pretrained Pix2Pix model for gray image colorization ############



os.environ["CUDA_VISIBLE_DEVICES"]= "1,2,3" 


def load_image(path):
    img = imread(path)
    # crop image from center
    short_edge = min(img.shape[:2])
    yy = int((img.shape[0] - short_edge) / 2)
    xx = int((img.shape[1] - short_edge) / 2)
    crop_img = img[yy : yy + short_edge, xx : xx + short_edge]
    # resize to 224, 224
    img = skimage.transform.resize(crop_img, (224, 224))
    # desaturate image
    return (img[:,:,0] + img[:,:,1] + img[:,:,2]) / 3.0

path = 'dataset'
images = []
for dirName,subDir,filelist in sorted(os.walk(path)):
    for filename in filelist:
        if '.png' in filename:
            images.append(os.path.join(dirName,filename))
        if '.jpg' in filename:
            images.append(os.path.join(dirName,filename))
        if '.jpeg' in filename:
            images.append(os.path.join(dirName,filename))

with open("model/colorize.tfmodel", mode='rb') as f:
    fileContent = f.read()

graph_def = tf.GraphDef()
graph_def.ParseFromString(fileContent)
grayscale = tf.placeholder("float", [1, 224, 224, 1])
tf.import_graph_def(graph_def, input_map={ "grayscale": grayscale }, name='')

with tf.device('/gpu:1'): ### using first GPU
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)) as sess:
        inferred_rgb = sess.graph.get_tensor_by_name("inferred_rgb:0")
        try:
            for img in images:
                print('image ::::::::: ',img)
                gray_img = load_image(img).reshape(1, 224, 224, 1)
                inferred_batch = sess.run(inferred_rgb, feed_dict={ grayscale: gray_img })
                imsave("colored_images/"+img.split('/')[-1], inferred_batch[0])
        except:
            print('mixture of images with 3 channnels and 1 channel found..')

print("images colored successfully!!!!!!")

