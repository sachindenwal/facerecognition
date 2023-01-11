import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from numpy import reshape
from mtcnn.mtcnn import MTCNN
from PIL import Image
from keras.models import load_model
from sklearn.svm import SVC
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

def face_extraction(image):
  img = plt.imread(image)
  detector = MTCNN()
  face_detection = detector.detect_faces(img)
  x1,y1,width,height = face_detection[0]['box']
  x1, y1 = abs(x1), abs(y1)
  x2, y2 = x1 + width, y1 + height
  face = img[y1:y2,x1:x2]
  face1 = Image.fromarray(face,'RGB')
  face1 = face1.resize((160,160))
  face2 = np.asarray(face1)
  return face2

def feature_embeddings(model,face_img):
  face_img = face_img.astype('float32')
  pixels_mean = face_img.mean()
  pixels_std  = face_img.std()
  face_img = (face_img - pixels_mean)/pixels_std
  features = np.expand_dims(face_img,axis=0)
  prediction = model.predict(features)
  return prediction[0]
  
def main():
  image = input("Enter file path for jpg image")
  face = face_extraction(image)
  test_x = np.asarray(face)
  test_x = test_x.reshape(-1,160,160,3)

  model = load_model("/content/drive/My Drive/Colab Notebooks/CS419 project/Facenet_keras.h5")
  newtest_x = list()
  for test_pixels in test_x:
    embeddings = feature_embeddings(model,test_pixels)
    newtest_x.append(embeddings)
  newtest_x = np.asarray(newtest_x)  
  print("Embedding dimension: ",newtest_x.shape[1])

  data = np.load("/content/drive/My Drive/Colab Notebooks/CS419 project/bollywood_celebs_embeddings.npz")
  train_x = data['arr_0']
  train_y = data['arr_1']

  input_encode = Normalizer(norm='l2')
  train_x = input_encode.transform(train_x)
  newtest_x = input_encode.transform(newtest_x)

  newtest_y = train_y 
  output_encode = LabelEncoder()
  output_encode.fit(train_y)
  train_y = output_encode.transform(train_y)
  newtest_y = output_encode.transform(newtest_y)

  model = SVC(kernel='linear', probability=True)
  model.fit(train_x,train_y)

  train_prediction = model.predict(train_x)
  test_prediction = model.predict(newtest_x)
# Results
  # trainY = output_encode.inverse_transform(train_y)
  # matrix = classification_report(train_y, train_prediction, labels=list(trainY))
  # print('Classification report: \n', matrix)
  print("accuracy of model for training data is", accuracy_score(train_y,train_prediction))
  confidence_score = np.max(model.predict_proba(newtest_x))
  print("Confidence score for test image's label prediction is", confidence_score)

  plt.plot()
  img = plt.imread(image)
  plt.imshow(img)
  axes = plt.gca()
  detector = MTCNN()
  face_detection = detector.detect_faces(img)
  x1,y1,width,height = face_detection[0]['box']
  traced_rectangle = Rectangle((x1, y1), width, height, fill=False, color='red')
  axes.add_patch(traced_rectangle)
  test_prediction = output_encode.inverse_transform(test_prediction)
  plt.title(test_prediction)
  plt.xlabel("Prediction of label of Input data")
  plt.show()

main()