from __future__ import print_function
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Reshape, Dense, Dropout, MaxPooling2D, Conv2D, Flatten
from tensorflow.keras.layers import Conv2DTranspose, LeakyReLU
from tensorflow.keras.layers import Activation#.core
from tensorflow.keras.layers import BatchNormalization #.normalization
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras import backend as K
from tensorflow.keras import initializers
import tensorflow as tf
import numpy as np
from tqdm import tqdm
import cv2
import math
import tensorflow as tf

####GPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
  try:
    tf.config.set_logical_device_configuration(
        gpus[0],
        [tf.config.LogicalDeviceConfiguration(memory_limit=4096)])
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Virtual devices must be set before GPUs have been initialized
    print(e)
from  tensorflow.keras.utils import Progbar #. generic_utils
###
### combine images for visualization
def combine_images(generated_images):
    num = generated_images.shape[0]
    width = int(math.sqrt(num))
    height = int(math.ceil(float(num)/width))
    shape = generated_images.shape[1:4]
    image = np.zeros((height*shape[0], width*shape[1], shape[2]),
                     dtype=generated_images.dtype)
    for index, img in enumerate(generated_images):
        i = int(index/width)
        j = index % width
        image[i*shape[0]:(i+1)*shape[0], j*shape[1]:(j+1)*shape[1],:] = img[:, :, :]
    return image

### generator300 model define
def generator_model():
    inputs = Input((10,))
    #fc1 = Dense(input_dim=10, units=128*7*7)(inputs)
    fc1 = Dense(input_dim=10, units=128*75*75)(inputs)
    fc1 = BatchNormalization()(fc1)
    fc1 = LeakyReLU(0.2)(fc1)
    #fc2 = Reshape((7, 7, 128), input_shape=(128*7*7,))(fc1)
    fc2 = Reshape((75, 75, 128), input_shape=(128*75*75,))(fc1)
    up1 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(fc2)
    conv1 = Conv2D(64, (3, 3), padding='same')(up1)
    conv1 = BatchNormalization()(conv1)
    conv1 = Activation('relu')(conv1)
    up2 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv1)
    conv2 = Conv2D(1, (5, 5), padding='same')(up2)
    outputs = Activation('tanh')(conv2)

    print("conv2_outputs_shape :",conv2.get_shape())

    model = Model(inputs=[inputs], outputs=[outputs])
    return model

### discriminator300 model define
def discriminator_model():
    # inputs = Input((28, 28, 1))
    inputs = Input((300, 300, 1))
    conv1 = Conv2D(64, (5, 5), padding='same')(inputs)
    conv1 = LeakyReLU(0.2)(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, (5, 5), padding='same')(pool1)
    conv2 = LeakyReLU(0.2)(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    #
    conv3 = Conv2D(64, (3,3), padding='same')(pool2)
    conv3 = LeakyReLU(0.2)(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(64, (3,3), padding='same')(pool3)
    conv4 = LeakyReLU(0.2)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    #
    fc1 = Flatten()(pool4)
    fc1 = Dense(1)(fc1)
    outputs = Activation('sigmoid')(fc1)
    print("fc1_outputs_shape :",fc1.get_shape())
    model = Model(inputs=[inputs], outputs=[outputs])
    return model

### d_on_g model for training generator300
def generator_containing_discriminator(g, d):
    d.trainable = False
    ganInput = Input(shape=(10,))
    x = g(ganInput)
    ganOutput = d(x)
    gan = Model(inputs=ganInput, outputs=ganOutput)
    # gan.compile(loss='binary_crossentropy', optimizer='adam')
    return gan

def load_model():
    d = discriminator_model()
    g = generator_model()
    d_optim = Adam(lr=0.00002)#RMSprop()
    g_optim = Adam(lr=0.00004)#RMSprop(lr=0.0002)
    g.compile(loss='binary_crossentropy', optimizer=g_optim)
    d.compile(loss='binary_crossentropy', optimizer=d_optim)
    d.load_weights('./weights/discriminator_300_WT0819.h5')
    g.load_weights('./weights/generator_300_WT8019.h5')
    return g, d
gg=[]
### train generator300 and discriminator300
def train(BATCH_SIZE, X_train):
    
    ### model define
    d = discriminator_model()
    g = generator_model()
    d_on_g = generator_containing_discriminator(g, d)
    #d_optim = RMSprop(lr=0.0004)
    #g_optim = RMSprop(lr=0.0002)
    d_optim = Adam(lr=0.00002)# lr=0.00008
    g_optim = Adam(lr=0.00004)
    g.compile(loss='mse', optimizer=g_optim)
    d_on_g.compile(loss='mse', optimizer=g_optim)
    d.trainable = True
    d.compile(loss='mse', optimizer=d_optim)
    

    
    for epoch in tqdm(range(750)):#650
        print ("Epoch is", epoch)
        n_iter = int(X_train.shape[0]/BATCH_SIZE)
        progress_bar = Progbar(target=n_iter)
        
        for index in range(n_iter):
            # create random noise -> U(0,1) 10 latent vectors
            noise = np.random.uniform(0, 1, size=(BATCH_SIZE, 10))

            # load real data & generate fake data
            image_batch = X_train[index*BATCH_SIZE:(index+1)*BATCH_SIZE]
            generated_images = g.predict(noise, verbose=0)
            # visualize training results
            if index % 5000 == 0:
                image = combine_images(generated_images)
                image = image*127.5+127.5
                cv2.imwrite('./result/'+str(epoch)+"_"+str(index)+".png", image)

            # attach label for training discriminator300
            X = np.concatenate((image_batch, generated_images))
            y = np.array([1] * BATCH_SIZE + [0] * BATCH_SIZE)
            
            # training discriminator300
            d_loss = d.train_on_batch(X, y)

            # training generator300
            d.trainable = False
            g_loss = d_on_g.train_on_batch(noise, np.array([1] * BATCH_SIZE))
            d.trainable = True
            
            progress_bar.update(index, values=[('g',g_loss), ('d',d_loss)])
            
            gg.append(g_loss)
            
        if epoch>=200:
            #print ('')
            #print(gg)
            #print ('')
            #print((gg[-1] == gg[-2]))
            if ((gg[-1] == gg[-2])==True):
                # save weights for each epoch
                g.save_weights('weights/generator_300_WT8019.h5', True)
                d.save_weights('weights/discriminator_300_WT0819.h5', True)
                return d, g
                #break
                
           
                
        print ('')

        # save weights for each epoch
        g.save_weights('weights/generator_300_WT8019.h5', True)
        d.save_weights('weights/discriminator_300_WT0819.h5', True)
    return d, g

### generate images
def generate(BATCH_SIZE):
    g = generator_model()
    g.load_weights('weights/generator_300_WT8019.h5')
    noise = np.random.uniform(0, 1, (BATCH_SIZE, 10))
    generated_images = g.predict(noise)
    return generated_images

### anomaly loss function 
def sum_of_residual(y_true, y_pred):
    return K.sum(K.abs(y_true - y_pred))

### discriminator300 intermediate layer feautre extraction
def feature_extractor(d=None):
    if d is None:
        d = discriminator_model()
        d.load_weights('weights/discriminator_300_WT0819.h5') 
    intermidiate_model = Model(inputs=d.layers[0].input, outputs=d.layers[-7].output)
    intermidiate_model.compile(loss='binary_crossentropy', optimizer='rmsprop')
    return intermidiate_model

### anomaly detection model define
def anomaly_detector(g=None, d=None):
    if g is None:
        g = generator_model()
        g.load_weights('weights/generator_300_WT8019.h5')
    intermidiate_model = feature_extractor(d)
    intermidiate_model.trainable = False
    g = Model(inputs=g.layers[1].input, outputs=g.layers[-1].output)
    g.trainable = False
    # Input layer cann't be trained. Add new layer as same size & same distribution
    aInput = Input(shape=(10,))
    gInput = Dense((10), trainable=True)(aInput)
    gInput = Activation('sigmoid')(gInput)
    
    # G & D feature
    G_out = g(gInput)
    D_out= intermidiate_model(G_out)    
    model = Model(inputs=aInput, outputs=[G_out, D_out])
    model.compile(loss=sum_of_residual, loss_weights= [0.90, 0.10], optimizer='rmsprop')
    
    # batchnorm learning phase fixed (test) : make non trainable
    K.set_learning_phase(0)
    
    return model

### anomaly detection
def compute_anomaly_score(model, x, iterations=500, d=None):
    z = np.random.uniform(0, 1, size=(1, 10))
    
    intermidiate_model = feature_extractor(d)
    d_x = intermidiate_model.predict(x)

    # learning for changing latent
    loss = model.fit(z, [x, d_x], batch_size=1, epochs=iterations, verbose=0)
    similar_data, _ = model.predict(z)
    
    loss = loss.history['loss'][-1]
    
    return loss, similar_data
