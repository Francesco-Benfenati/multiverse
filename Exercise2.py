from keras.models import Sequential
from keras.layers import Input, Dense, Dropout, Activation, Flatten
from keras.models import Model
from keras.optimizers import SGD
from keras.datasets import mnist
from keras.applications.vgg16 import VGG16
import numpy as np
import cv2
from datetime import datetime

tBatchSize = 64
#VGG16 requires a minimum image size of 48 
model_vgg = VGG16(include_top=False, input_shape=(48,48,3))
for layer in model_vgg.layers:
    layer.trainable = False

model = Flatten(name='flatten')(model_vgg.output)
model = Dense(500, activation='relu', name='fc1')(model)
model = Dense(500, activation='relu', name='fc2')(model)
model = Dropout(0.5)(model)
model = Dense(10, activation='softmax')(model)
model = Model(inputs=model_vgg.input, 
                        outputs = model, name = 'vgg16')
 
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

(X_train, y_train), (X_test, y_test) = mnist.load_data() # Read data using the mnist tool that comes with Keras
 
# Since the input data dimension of mist is (num, 28, 28), vgg16 needs a three-dimensional image, because the last dimension of mnist is expanded

X_train = [cv2.cvtColor(cv2.resize(i, (48, 48)), cv2.COLOR_GRAY2RGB) for i in X_train]
X_test = [cv2.cvtColor(cv2.resize(i, (48, 48)), cv2.COLOR_GRAY2RGB) for i in X_test]

X_train = np.array(X_train)
X_test = np.array(X_test)

# Generate a OneHot 10D vector as a line of Y_train: Y_train has 60,000 lines of OneHot as output
Y_train = (np.arange(10) == y_train[:, None]).astype(int) #  
Y_test = (np.arange(10) == y_test[:, None]).astype(int)   
 
# Normalize to [0,1]
X_train = X_train/255
X_test = X_test/255

start = datetime.now()
history = model.fit(X_train, Y_train, batch_size=tBatchSize, epochs=5, shuffle=True, validation_split=0.3)
duration = datetime.now() - start
print("Training completed in time: ", duration)

score = model.evaluate(X_test, Y_test, batch_size=tBatchSize)
print('Test Loss:', score[0])
print('Test Accuracy:', score[1])

import matplotlib.pyplot as plt
plt.plot(history.history["accuracy"])
plt.plot(history.history['val_accuracy'])
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title("Model accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Epoch")
plt.legend(["Accuracy","Validation Accuracy","Loss","Validation Loss"])
plt.show()