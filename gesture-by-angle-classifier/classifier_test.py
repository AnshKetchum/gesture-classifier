'''
This script was built to test the GestureAngleClassifier (angle based NN)
and ensure that the prediction function is operations.
'''
from models import GestureAngleClassifier

#Create the model
model = GestureAngleClassifier('train_angles.npy', 'train_labels.npy')
print('Model Sucessfully Built.')


#Get the prediction - 0th index = classification, 1st index - similarly
pred = model.get_predictions('../gesture-by-feature-classifier/Dataset/Backhand/Image-5537.jpg')
print(pred[0])