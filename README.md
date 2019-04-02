# Neural-Nebula
A deep convolutional generative adversarial network (DCGAN) is trained on pictures of space. Images can be procedurally created from the generative neural network by sampling the latent space. Information on the neural network can be found here: https://arxiv.org/abs/1511.06434

## Dependencies
- [Python 3.6+](https://www.anaconda.com/distribution/)
- Keras, Tensorflow==1.4, Matplotlib, Numpy, PIL, Scikit-learn

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
Below is an animation of the training process using 10001 training batches which took ~10 minutes on a GTX 1070
![](https://github.com/pearsonkyle/Neural-Nebula/blob/master/images/cifar_bird.gif)

These are random samples from the generator during training. After just 10 minutes of training you can start to see structure that resembles a bird. There's only so much structure you can get from a 32 x 32 pixel image to begin with... More realistic images can be chosen by evaluating them with the discriminator after generating. 


## Creating a custom data set
The  `create_dataset` function will cut random slices from an images to create a new data set. This function requires you to put images a new directory before hand
```python
from dcgan import create_dataset 

# first resize the original image to 75% 
# then cut 100 random 128x128 subframes from the image 
x_train, y_train = create_dataset(128,128, nSlices=100, resize=0.75, directory='images/')

# scale RGB data between 0 and 1
x_train /= 255 
```

## Higher Resolution Images 
If you want to produce data sets at a resolution higher than 32x32 pixels you will have to modify the architecture of the GAN yourself. For example, uncommenting the two UpSampling2D() functions in build_generator() will increase the size of the images to 128x128. Consequently, the discriminator can be modified in the build_discriminator() function. 


## Exporting a model to Unity
The reason I quoted tensorflow v1.4 in the dependencies is to match the version of tensorflow in UnityML just incase you want to upload these models into Unity

- export_model
- load into unity
- attach mesh generator script 
