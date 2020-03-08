from models import GestureAngleClassifier

model = GestureAngleClassifier('train_angles.npy', 'train_labels.npy')
print('Model Sucessfully Built.')
