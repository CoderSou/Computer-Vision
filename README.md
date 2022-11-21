# Computer-Vision
Advanced deep learning based computer vision systems within the Python/Keras environment.
A deep learning based system to classify 6 food categories from the Food-101 data set. 

The 6 food classes of interest are Chicken Curry, Hamburger, Omelette, Pancakes, Spaghetti Bolognese and Waffles.
For each class, 150 manually reviewed validation images, 100 manually reviewed test images are provided as well as 750 training images.
Data check is performed, and images are augmented to increase model robustness.

Firstly, a simple working basline convolutional neural network architecture to classify the 6 food classes is implemented.
Secondly, a highly accurate and computationally efficient solution using pre-trained Xception model and transfer learning is used to fine tune the pre-trained network for this specific application. 
Dropouts are used for overfitting conditions, Activation ReLU is used to introduce some non-linearity and then sigmoid was used to get the probabilities.
