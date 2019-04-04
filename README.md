# Neural-Nebula
A deep convolutional generative adversarial network (DCGAN) is trained on pictures of space. Images can be procedurally created from the generative neural network by sampling the latent space. Information on the neural network architecture can be found here: https://arxiv.org/abs/1511.06434

![](https://github.com/pearsonkyle/Neural-Nebula/blob/master/nebula.gif)

## Dependencies
- [Python 3.6+](https://www.anaconda.com/distribution/)
- Keras, Tensorflow, Matplotlib, Numpy, PIL, Scikit-learn

## Example
Clone the repo, cd into the directory, launch iPython and paste the example below 
```python 
from keras.datasets import cifar10
from dcgan import DCGAN

if __name__ == '__main__':

    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # only birds, then scale images between 0-1
    x_train = x_train[ (y_train==2).reshape(-1) ] 
    x_train = x_train/255
    
    dcgan = DCGAN(img_rows = x_train[0].shape[0],
                    img_cols = x_train[0].shape[1],
                    channels = x_train[0].shape[2], 
                    latent_dim=128,
                    name='cifar_128')

    dcgan.train(x_train, epochs=10001, batch_size=32, save_interval=100)
    
    dcgan.save_imgs('final') 
```
Below is an animation of the training process every 100 training batches. The code above took ~10 minutes to run on a GTX 1070. These are random samples from the generator during training. After just 10 minutes of training you can start to see structure that resembles a bird. There's only so much structure you can get from a 32 x 32 pixel image to begin with... More realistic images can be chosen by evaluating them with the discriminator after generating. 

![](https://github.com/pearsonkyle/Neural-Nebula/blob/master/images/cifar_bird.gif)

## Creating a custom data set
The  `create_dataset` function will cut random slices from an images to create a new data set. This function requires you to put images a new directory before hand
```python
import matplotlib.pyplot as plt
import numpy as np

from dcgan import create_dataset 

# first resize the original image to 75% 
# then cut 100 random 128x128 subframes from each image in the directory 
x_train, y_train = create_dataset(128,128, nSlices=100, resize=0.75, directory='space/')

# scale RGB data between 0 and 1
x_train /= 255 

# plot results to make sure data looks good!
fig, axs = plt.subplots(4, 4)
for i in range(4):
    for j in range(4):
        axs[i,j].imshow( x_train[ np.random.randint(x_train.shape[0]) ] )
        axs[i,j].axis('off')
plt.show()
```
An example output should look like this: 

![](https://github.com/pearsonkyle/Neural-Nebula/blob/master/images/nebula_training_sample.png)

If `x_train` is empty make sure you have .jpg or .png files in the directory where your images are stored (e.g. space/) 


## Higher Resolution Images 
If you want to produce data sets at a resolution higher than 32x32 pixels you will have to modify the architecture of the GAN yourself. For example, uncommenting the two `UpSampling2D()` functions in `build_generator()` will increase the size of the images to 128x128.

## Sampling the latent space
Use the generator, for an example see the (`save_imgs()`)[] method

## Animating the training steps
check the directory images/ and then use Imagemagick or ffmpeg to create a gif 

`ffmpeg -framerate 60 -i "nebula_%05d.png" -i "planet_caravan.mp3" -map 0:v:0 -map 1:a:0 -c:v libx264 -pix_fmt yuv420p -strict -2 -shortest nebula.mp4`
