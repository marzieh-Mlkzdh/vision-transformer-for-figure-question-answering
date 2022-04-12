import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.layers import Input, Dense
from tensorflow.python.keras.models import Sequential
from keras import optimizers

# To train our model, first it is necessary to merge the image features and question features together while paying attention to image_index , so now we have
# an image which is repeated in different rows with new questions each time.


qa_dataFrame = pd.read_csv('qa_pairs.csv')
qa_embeddings = pd.read_csv('QuestionFeatures.csv')
image_embeddings = pd.read_csv('imageFeatures.csv')




numbers = qa_dataFrame["image_index"]
answers = qa_dataFrame["answer"]
qa_embeddings = qa_embeddings.join(numbers)
qa_embeddings = qa_embeddings.join(answers)
image_embeddings['image_index'] = image_embeddings.index
df_tot = pd.merge(qa_embeddings, image_embeddings, on='image_index', how='inner')


X = df_tot
Y = df_tot[["answer"]]

X_train, X_test, y_train, y_test = train_test_split(df_tot, df_tot[["answer"]], test_size=0.1, random_state=43)


model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(df_tot.shape[1],)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))




model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
history = model.fit(X_train, y_train, epochs=500, batch_size=64, validation_data=(X_test, y_test))

model.save('path/to/location')

