import tensorflow as tgf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import mathplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

CUBE_COLORS = ['white', 'yellow', 'red', 'orange', 'blue', 'green']

# load images function
def load_image_folder(folder):
  images = []
  labels = []
  
  for color in CUBE_COLORS:
    path = os.path.join(folder, color)

    if not os.path.exists(path):
      print("Folder was not found.")
      continue

  for img_name in os.listdir(path, img_name):
    img_path = os.path.join (path, img_name)
    img = cv2.resize(32, 32)
    img = img / 255.0
    images.append(img)
    labels.append(CUBE_COLORS.index(color))
  
  return np.array(imgaes), np.array(labels)


data_set_path = 'will do add this later lol'
x, y = load_images_folder(data_set_path)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)

# CNN MODEL
model = keras.sequential( [
  layers.Conv2D(32, (3,3), activation='relu', input_shape=(32, 32, 3)),
  layers.MaxPooling2D(2,2)),
  layers.Conv2D(64, (3,3), activation='relu'),
  layers.MaxPooing2D(2,2),
  layers.Flatten(), 
  layers.Dense(64, activation='relu'),
  layers.Dense(len(CUBE_COLORS), activation='softmax'
])

# Compiling model

model.compile(optimize='adam', loss='sparse_categorical_crossentropy', metrix=['accuracy'])

# training model
history = model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))

# saving model 
model.save('rubix_cube_color_model.h5')

loss, acc = model.evaluate(x_test, y_test)
print(f'Test Accuracy: {acc * 100: 2f}%')

#plotting training history








                
