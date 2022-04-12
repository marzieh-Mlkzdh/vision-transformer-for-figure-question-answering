from transformers import ViTFeatureExtractor, ViTModel
from PIL import Image
import requests
import glob
import numpy as np
from numpy import savetxt
import pandas as pd



# To Download the Vision Transformer Feature Extractor
feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')



# if using the SBU-FQA dataset, no need to change the folders' name.
for filename in glob.glob('/content/drive/MyDrive/train2/*.png'):
    im=Image.open(filename)
    im= im.convert('RGB')
    image_list.append(im)




# To Download the ViT pretrained model
model = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')




features_list = np.zeros((len(image_list) , 197 , 768))
for i in range(len(image_list)):
    inputs = feature_extractor(images=image_list[i], return_tensors="pt")
    outputs = model(**inputs)
    last_hidden_states = outputs.last_hidden_state
    features_list[i] = last_hidden_states.detach().numpy()
    
features_list = features_list.reshape(len(image_list), (197*768))



np.savetxt('/content/drive/MyDrive/imageFeatures.csv', features_list, delimiter=',')

