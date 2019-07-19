import cv2
import matplotlib.pyplot as plt

img_array=cv2.imread('frame67.jpg')
plt.imshow(img_array,cmap="gray")
plt.show()
img_size=80
new_array=cv2.resize(img_array,(img_size,img_size))
plt.imshow(new_array,cmap='gray')
plt.show()
'''
from keras.models import load_model
import cv2
import numpy as np

model = load_model('model.h5')

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

img_data = cv2.imread(r'E:\Nikhil\python\walking.jpg',cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img,(100,100))
img = np.reshape(img,[1,100,100,1])

classes = model.predict_classes(img)

print classes
'''
