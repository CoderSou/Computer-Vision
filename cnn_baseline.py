"""
Designing a simple, baseline convolutional neural network architecture to classify the 6 food classes.
"""
# Import necessary libraries
import os
from time import time
import matplotlib.pyplot as plt
from keras import layers
from keras import models
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.convolutional import ZeroPadding2D

# Load Data
base_dir = 'C:/Users/Desktop/food101_selected/'
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')
test_dir = os.path.join(base_dir, 'test')

#CNN network; Convolutional, Padding, Pooling, Flatten, Dense layers
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), strides=(1, 1), activation='relu',input_shape=(200, 200, 3)))
model.add(ZeroPadding2D((1,1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3),strides=(1, 1), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3),strides=(1, 1), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(256, (3, 3),strides=(1, 1), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(ZeroPadding2D((1,1)))
model.add(layers.Conv2D(512, (3, 3),strides=(1, 1), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(1024, activation='relu'))
model.add(layers.Dense(6, activation='sigmoid'))

#Displaying a summary of the model
print(model.summary())

#Start of timer
t0 = time()

#Normalize
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)
validation_datagen = ImageDataGenerator(rescale=1./255)


train_generator = train_datagen.flow_from_directory(train_dir, target_size=(200, 200), batch_size=30, class_mode='categorical')
validation_generator = validation_datagen.flow_from_directory(validation_dir, target_size=(200, 200), batch_size=20, class_mode='categorical')

model.compile(loss='categorical_crossentropy', optimizer=optimizers.RMSprop(lr=1e-4), metrics=['accuracy'])

# Fit the model
history = model.fit_generator(train_generator, steps_per_epoch=150, epochs=20,validation_data=validation_generator, validation_steps=45)
# Save the model
model.save('final_base.h5')

test_generator = test_datagen.flow_from_directory(test_dir, target_size=(200, 200), batch_size=20, class_mode='categorical')

#Evaluation of test, train and validation
results_test = model.evaluate_generator(test_generator, steps=30)
result_train = model.evaluate_generator(train_generator, steps=150)
results_validate = model.evaluate_generator(validation_generator, steps=45)

# Printing Results
print('Final test accuracy:', (results_test[1]*100.0))
print('Final train accuracy:', (result_train[1]*100.0))
print('Final validate accuracy:', (results_validate[1]*100.0))
print('CNN Error: ', (100 - (results_test[1] * 100.0)))
#Computational Cost in seconds
print('final model took', int(time() - t0), 's')

# Plot accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Enhanced CNN model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train','validation'], loc='upper left')
plt.savefig('C:/Users/Guest/Desktop/New folder/results/base_1.jpg', dpi=250)
print('Close images to continue')
plt.show()

# Plot loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Enhanced CNN model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.savefig('C:/Users/Guest/Desktop/New folder/results/base_2.jpg', dpi=250)
print('Close images to continue')
plt.show()
