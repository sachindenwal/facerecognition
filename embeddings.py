from os import listdir
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from mtcnn.mtcnn import MTCNN
from keras.models import load_model

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

def load_faces(folder_directory):
  faces = []
  i=1
  for filename in listdir(folder_directory):
    image_path = folder_directory + filename
    face = face_extraction(image_path)
    faces.append(face)
  return faces

def load_dataset(dataset_directory):
  x, y = [],[]
  i=1
  for folder_directory in listdir(dataset_directory):
    path = dataset_directory + folder_directory + '/'
    faces = load_faces(path)
    names = [folder_directory for _ in range(len(faces))]
    x.extend(faces)
    y.extend(names)
    i=i+1
  return np.asarray(x),np.asarray(y)

def feature_embeddings(model,face_img):
  face_img = face_img.astype('float32')
  pixels_mean = face_img.mean()
  pixels_std  = face_img.std()
  face_img = (face_img - pixels_mean)/pixels_std
  features = np.expand_dims(face_img,axis=0)
  prediction = model.predict(features)
  return prediction[0]

def main():
    global val
    train_X,train_Y = load_dataset("/content/drive/My Drive/Colab Notebooks/CS419 project//")
    model = load_model("/content/drive/My Drive/Colab Notebooks/CS419 project/Facenet_keras.h5")

    newtrain_X = list()
    for train_img in train_X:
        embeddings = feature_embeddings(model, train_img)
        newtrain_X.append(embeddings)
    newtrain_X = np.asarray(newtrain_X)
    print(newtrain_X.shape)
    np.savez_compressed('/content/drive/My Drive/Colab Notebooks/CS419 project/bollywood_celebs_embeddings.npz',newtrain_X,train_Y)

main()
