import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

model = tf.keras.models.load_model('fruit_classifier_xception.h5')

class_names = ['Apple', 'Banana', 'Grape', 'Mango', 'Strawberry']

def predict_fruit(image_path):
    img = image.load_img(image_path, target_size=(299, 299))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions[0])
    predicted_fruit = class_names[predicted_class]
    confidence = predictions[0][predicted_class]

    return predicted_fruit, confidence

image_path = 'path_to_your_test_image.jpg'
predicted_fruit, confidence = predict_fruit(image_path)
print(f"Predicted fruit: {predicted_fruit} with confidence: {confidence:.2f}")
