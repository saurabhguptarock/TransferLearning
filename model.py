from tensorflow.contrib import lite
from keras.models import Model
from keras.layers import GlobalAveragePooling2D, Dense, Dropout
from keras.optimizers import Adam
import numpy as np
from keras.preprocessing import image
from keras.applications.mobilenet_v2 import MobileNetV2
import matplotlib.pyplot as plt
from keras.utils.np_utils import to_categorical

from keras.datasets import mnist
data = mnist.load_data()

(xtrain, ytrain), (xtest, ytest) = data
ytrain = to_categorical(ytrain)

demo = np.zeros((60000, 96, 96))
x_new_train = np.zeros((60000, 96, 96, 3))

# Convert 28x28 Image to 96x96
for i in range(xtrain.shape[0]):
    demo[i] = np.pad(xtrain[i], 34, mode='constant')
    x_new_train[i] = np.stack((demo[i], demo[i], demo[i]), axis=2)

# Initialize MobileNet Model
model = MobileNetV2(weights='imagenet', include_top=False,
                    input_shape=(96, 96, 3))

# New Model.
av1 = GlobalAveragePooling2D()(model.output)
fc1 = Dense(256, activation='relu')(av1)
d1 = Dropout(0.5)(fc1)
fc2 = Dense(10, activation='softmax')(d1)

model_new = Model(input=model.input, output=fc2)

for ix in range(151):
    model_new.layers[ix].trainable = False

model_new.compile(loss='categorical_crossentropy',
                  optimizer=Adam(0.0003), metrics=['accuracy'])

hist = model_new.fit(x_new_train,  ytrain, shuffle=True,
                     batch_size=256, epochs=20)
model_new.save('mobilenet_v2_224_quant.h5')


# Convert model to TF Lite
converter = lite.TFLiteConverter.from_keras_model_file(
    'mobilenet_v2_224_quant.h5')
tfmodel = converter.convert()
open("mobilenet_v2_224_quant.tflite", "wb") .write(tfmodel)
