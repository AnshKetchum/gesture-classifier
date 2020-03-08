from models import GestureAngleClassifier

model = GestureAngleClassifier('train_angles.npy', 'train_labels.npy')
print('Model Sucessfully Built.')

pred = model.get_predictions('../gesture-by-feature-classifier/Dataset/Backhand/Image-5537.jpg')
print(pred[0])