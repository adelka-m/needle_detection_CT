
import numpy as np
import tensorflow.keras.backend as K
import segmentation_models as sm
sm.set_framework('tf.keras')


class DataGenerator(keras.utils.Sequence):
    #'Generates data for Keras'
    def __init__(self, list_IDs, path, batch_size=32, dim=(512,512), n_channels=3, 
                  thr = None, thrmax = None,
                  augment = False, shuffle=True):
     
        #'Initialization'                    
        self.dim = dim
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.thr = thr
        self.thrmax = thrmax
        self.augment = augment
        self.shuffle = shuffle
        self.path = path
        self.on_epoch_end()

    def __len__(self):
        #'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        #'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        #'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        #'Generates data containing batch_size samples' 
        # X : (n_samples, *dim, n_channels)

        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size, *self.dim, 1))

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            
            img = np.expand_dims(np.load(self.path + str(ID[-2]) + '/scan_needles_' + str(ID[-1]).zfill(3) + '.npy'), axis = -1)
            needles = np.load(self.path + str(ID[-2]) + '/needles_' + str(ID[-1]).zfill(3) + '.npy', allow_pickle=True)

            # create needle mask img
            y_help = np.zeros((512,512,1))
            if len(needles) > 0:
                for needle in needles:
                    y_help[np.transpose(needle)[0],np.transpose(needle)[1]] = 1  

            # preprocess imgs
            if self.dim != (512,512):
                img = resize(img, (self.dim), anti_aliasing=True)
                y_help = (resize(y_help, self.dim, anti_aliasing=False) > 0).astype(int)
                      
            # threshold
            if self.thr != None:
                img[img<self.thr] = self.thr
            if self.thrmax != None:
                img[img>self.thrmax] = self.thrmax
            
             # random augmentations
            if self.augment:
                # generate random subsample of augmentations
                n_aug = 6
                id_aug = np.random.choice(np.arange(n_aug + 1), np.random.randint(n_aug + 1))
                if 1 in id_aug:
                    img = np.flip(img, axis = 1)
                    y_help = np.flip(y_help, axis = 1)
                if 2 in id_aug:
                    img = random_noise(img, mode = 'speckle', clip=False)
                if 3 in id_aug:
                    img = random_noise(img, mode = 'gaussian', clip=False)
                if 4 in id_aug:
                    img = random_noise(img, mode = 's&p', clip=False)
                if 5 in id_aug:
                    rot = np.random.choice([-5,5,15,-15], 1)[0]
                    img = rotate(img, rot, resize = False, mode = 'edge', clip = True)
                    y_help = rotate(y_help, rot, resize = False, mode = 'edge', clip = True) 
            
            # Image normalization to 0,255
            img = (img - np.min(img)) / np.max(img  - np.min(img))   * 255.0
            
            # Store sample
            data = np.zeros((*self.dim, 3))
            data[:,:,0] = np.squeeze(img) 
            data[:,:,1] = np.squeeze(img)
            data[:,:,2] = np.squeeze(img)
            X[i,] = data 
            y[i,] = y_help
        return X, y


def dice_loss(y_true, y_pred, smooth=1):        
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    dice = (2 * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return 1-dice


def norm_un(dim = (512, 512,1)):
    
    inputShape = dim
    chanDim = -1
    
    unet = sm.Unet('resnet18', encoder_weights='imagenet', classes=1, activation='sigmoid', encoder_freeze=True)
    preprocess_input = sm.get_preprocessing('resnet18')
    
    inputs = Input(shape= inputShape)
    x = preprocess_input(inputs)
    x = unet(x)
    x = Dropout(0.1)(x)
    
    model = Model(inputs = inputs, outputs = x, name='CNN')
    return model



def __main__():

	# load data and create training generator
	cwd = os.getcwd()
	path_0 = str(Path(cwd).parents[1])
	path = path_0 + '/YOUR-datapath/'
	# _: thick, 1: thin, 2: dilation no gauss mask

	df = pd.read_csv( path + '/csv_files/test.csv')
	test = df.values.tolist()
	df = pd.read_csv(path + '/csv_files/train.csv')
	train = df.values.tolist()

	n = 256
	dim = (n,n)
	thr = -200
	thrmax = 800

	training_generator = DataGenerator(train, dim = dim,  batch_size=8, thr = thr, thrmax = thrmax, augment = True, shuffle = True, path = path          )
	test_generator = DataGenerator(test, dim = dim, thr = thr, thrmax = thrmax, path = path)


	model = norm_un((*dim,3))
	model.compile(
	    opt,
	    loss=dice_loss,
	    metrics=[sm.metrics.iou_score],
	    )

	# fit model
	callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=20)
	H = model.fit(
	    training_generator, validation_data=test_generator,
	    epochs=150, #callbacks=[callback]
	    )














