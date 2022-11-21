"""
Developing a computationally efficient and highly accurate solution using transfer learning and
fine tuning concepts on baseline CNN architecture to optimise for the food classification task.
"""
#Import necessary libraries
import os
from time import time
import matplotlib.pyplot as plt
import keras
from keras import layers
from keras import models
from keras import optimizers
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator

#Load Data
base_dir = 'C:/Users/Guest/Desktop/New folder/food101_selected/'
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')
test_dir = os.path.join(base_dir, 'test')

# Transfer Learning - Xception is loaded
conv_base = keras.applications.xception.Xception(include_top=False, input_shape=(200, 200, 3), pooling='max', weights='imagenet', classes=6)
# All layers Freezed
conv_base.trainable = False
# Printing Summary of the Xception model
conv_base.summary()
# Our own Fully-Connected layer
model = models.Sequential()
model.add(conv_base)
model.add(layers.Dropout(0.5))
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(6, activation='softmax'))
# Summary of the entire model
model.summary()

# All layers un-freezed
conv_base.trainable = True
set_trainable = False
for layer in conv_base.layers:
    if layer.name == 'block14_sepconv1':      # All layers upto this layer are frezed
        set_trainable = True
    if set_trainable:
        layer.trainable = True
    else:
        layer.trainable = False

model = models.Sequential()
model.add(conv_base)
model.add(layers.Dropout(0.5))
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(6, activation='softmax'))
model.summary()

# Timer started
t0 = time()
# Data augmentation
train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=50, width_shift_range=0.3, height_shift_range=0.3, shear_range=0.3, zoom_range=0.3, vertical_flip=True, horizontal_flip=True, fill_mode='reflect')
test_datagen = ImageDataGenerator(rescale=1./255)
validation_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(train_dir, target_size=(200, 200), batch_size=30, class_mode='categorical')
validation_generator = validation_datagen.flow_from_directory(validation_dir, target_size=(200, 200), batch_size=20, class_mode='categorical')

# Compile full model
model.compile(loss='categorical_crossentropy', optimizer=optimizers.Adagrad(lr=0.01), metrics=['accuracy'])
history = model.fit_generator(train_generator, steps_per_epoch=150, epochs=20, validation_data=validation_generator, validation_steps=45, verbose=2)

# Save the model
model.save('final_version_1.h5')
test_generator = test_datagen.flow_from_directory(test_dir, target_size=(200, 200), batch_size=20, class_mode='categorical')

# Evaluate the model
results_test = model.evaluate_generator(test_generator, steps=30)
result_train = model.evaluate_generator(train_generator, steps=150)
results_validate = model.evaluate_generator(validation_generator, steps=45)

# Print accuracy for test, train, validation and CNN Error
print('Final test accuracy:', (results_test[1]*100.0))
print('Final train accuracy:', (result_train[1]*100.0))
print('Final validate accuracy:', (results_validate[1]*100.0))
print('CNN Error: ', (100 - results_test[1] * 100.0))

# Computational cost in minutes
print('final model took', (float(time() - t0)/60), 'min')

# Summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Enhanced CNN model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.savefig('C:/Users/Guest/Desktop/New folder/results/final_1.jpg', dpi=250)
print('Close images to continue')
plt.show()

# Summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Enhanced CNN model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.savefig('C:/Users/Guest/Desktop/New folder/results/final_2.jpg', dpi=250)
print('Close images to continue')
plt.show()
