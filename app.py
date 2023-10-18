import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model

ResNet50 = keras.applications.ResNet50(weights='imagenet', include_top=False)

model = load_model('auto_defect_detection.keras')

cap = cv2.VideoCapture(0)

class_labels = ['Minor Damage', 'Moderate Damage', 'Severe Damage'] 

while True: 
    ret, frame = cap.read()

    predictions = model.predict(np.expand_dims(frame, axis=0))

    # # Get the class with the highest probability
    predicted_class = np.argmax(predictions)

    # # Get the class label
    class_label = class_labels[predicted_class]

    # # Display the class label on the frame
    cv2.putText(frame, class_label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Display the frame with the class label
    cv2.imshow('Webcam Feed', frame)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

