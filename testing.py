# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing import image
from keras.models import load_model
from sklearn import preprocessing

# Path
img_path = 'C:/Users/Guest/Desktop/New folder/code/6432.jpg'
img = image.load_img(img_path, target_size=(200, 200))
image = image.img_to_array(img)

# Load prediction model
model = load_model('final_version.h5')
print('Model ready for prediction')
# Model summary
model.summary()

image = np.expand_dims(image, axis=0)
print(image.shape)
le = preprocessing.LabelEncoder()
le.fit(["Pancakes", "Hamburger", "Chicken Curry", "Omelette", "Spaghetti Bolognese", "Waffles"])

# Classification into the 6 food classes
print(le.classes_)
print("classifying image...")
image_class = model.predict_classes(image)
image_class = image_class.item(0)
print(image_class)
class_label = le.inverse_transform([image_class])
print(class_label)
