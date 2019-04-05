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
                    name='cifar10')

    dcgan.train(x_train, epochs=10001, batch_size=32, save_interval=500)