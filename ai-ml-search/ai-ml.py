########## Standard Import #########################################################################################
import sys
import os
sys.path.append(os.path.expanduser('~/Projects/bry_mod/'))
from bry_mod import *

########## Recomendation System (VIDEO REF) #########################################################################################
# User based recomendation system
import pandas as pd
import numpy as np
import scipy.stats
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity

ratings = pd.read_csv('/home/bry/Projects/study/ai-ml-search/files/ratings.csv')
ratings_head = ratings.head()
save(ratings_head)

movies = pd.read_csv('/home/bry/Projects/study/ai-ml-search/files/movies.csv')
movies_head = movies.head()
save(movies_head)

merged = pd.merge(ratings, movies, on='movieId', how='inner')
merged_head = merged.head()
save(merged_head)

agg_ratings = merged.groupby('title').agg(mean_rating=('rating', 'mean'), number_of_ratings=('rating', 'count')).reset_index()
save(agg_ratings)

agg_ratings_gt100 = agg_ratings[agg_ratings['number_of_ratings'] > 100 ]
save(agg_ratings_gt100)

agg_ratings_gt100.sort_values(by = 'number_of_ratings', ascending=False).head()
seaborn_plot = sns.jointplot(x='mean_rating', y='number_of_ratings', data=agg_ratings_gt100)
save(seaborn_plot)

df_gt100 = pd.merge(merged, agg_ratings_gt100[['title']], on='title', how='inner')
save(df_gt100)

matrix = df_gt100.pivot_table(index='userId', columns='title', values='rating')
save(matrix)

#ratings below the average are assigned a negative value while those above the average are given a positive value
matrix_norm = matrix.subtract(matrix.mean(axis=1), axis = 'rows')
save(matrix_norm)

# user similarity matrix
user_similarity = matrix_norm.T.corr()
save(user_similarity)

# inpute NaN with 0
user_similarity_cosine = cosine_similarity(matrix_norm.fillna(0))
save(user_similarity_cosine)

picked_userid = 1
user_similarity.drop(index=picked_userid, inplace=True)
save(user_similarity)

n = 10
user_similarity_threshold = 0.3
similar_users = user_similarity[user_similarity[picked_userid]>user_similarity_threshold][picked_userid]
save(similar_users)

########## Recomendation System #########################################################################################
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors

# Load dataset
ratings_data = pd.read_csv('/home/bry/Projects/study/ai-ml-search/files/ratings.csv', na_values='')

# Split into train and test sets
train_data, test_data = train_test_split(ratings_data, test_size=0.2)

# Collaborative filtering
user_item_matrix = train_data.pivot_table(index='userId', columns='movieId', values='rating').fillna(0)
user_similarity = cosine_similarity(user_item_matrix)
item_similarity = cosine_similarity(user_item_matrix.T)
save(user_similarity)
#save(item_similarity) !!! This file is too big and crashes system
print("Shape of user_similarity:", user_similarity.shape)
print("Shape of item_similarity:", item_similarity.shape)

# Predict ratings
def predict_rating(user_id, movie_id):
    user_ratings = user_item_matrix.loc[user_id].values.reshape(-1, 1)
    movie_ratings = user_item_matrix[movie_id].values.reshape(-1, 1)
    save(user_ratings)
    save(movie_ratings)
    user_similarities = user_similarity[user_id].reshape(-1, 1)
    movie_similarities = item_similarity[movie_id]
    print("Shape of user_similarities:", user_similarities.shape)
    print("Shape of user_ratings:", user_ratings.shape)
    save(user_ratings)
    save(movie_ratings)

    user_predicted_rating = user_similarities.dot(user_ratings) / sum(user_similarities)
    movie_predicted_rating = movie_similarities.dot(movie_ratings) / sum(movie_similarities)

    return (user_predicted_rating + movie_predicted_rating) / 2

# Example prediction
user_id = 1
movie_id = 1
predicted_rating = predict_rating(user_id, movie_id)
print(f"Predicted rating by user {user_id} for movie {movie_id}: {predicted_rating[0][0]}")


########## NLP ######################################################################################################
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import Counter

# Example text
text = "Natural language processing (NLP) is a subfield of linguistics, computer science, and artificial intelligence concerned with the interactions between computers and human language."

# Tokenization
tokens = word_tokenize(text.lower())

# Remove stopwords
stop_words = set(stopwords.words('english'))
filtered_tokens = [word for word in tokens if word.isalnum() and word not in stop_words]

# Word frequency
word_freq = Counter(filtered_tokens)
print("Word frequency:", word_freq)

# Part-of-speech tagging
pos_tags = nltk.pos_tag(tokens)
print("Part-of-speech tagging:", pos_tags)


########## Deep Learning ######################################################################################################
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten

# Example CNN model for image classification
model = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Example training data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.reshape((-1, 28, 28, 1)) / 255.0
x_test = x_test.reshape((-1, 28, 28, 1)) / 255.0

# Train the model
model.fit(x_train, y_train, epochs=5, batch_size=32, validation_data=(x_test, y_test))


########## Image Recognition ######################################################################################################
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
import numpy as np

# Load MobileNetV2 model pre-trained on ImageNet
model = MobileNetV2(weights='imagenet')

# Load and preprocess image
img_path = 'image.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

# Predict class probabilities
preds = model.predict(x)

# Decode predictions
decoded_preds = decode_predictions(preds, top=3)[0]

# Display top predictions
for i, (imagenet_id, label, score) in enumerate(decoded_preds):
    print(f"{i + 1}: {label} ({score:.2f})")


########## Machine Learning Models ######################################################################################################
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load dataset
iris = load_iris()
X, y = iris.data, iris.target

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Predictions
y_pred = clf.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


########## PyTorch ######################################################################################################
import torch
import torch.nn as nn
import torch.optim as optim

# Example PyTorch model
class SimpleNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Example usage
input_dim = 10
hidden_dim = 5
output_dim = 2
model = SimpleNN(input_dim, hidden_dim, output_dim)
print(model)


########## Tensor Flow ######################################################################################################
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Example TensorFlow model
model = Sequential([
    Dense(64, activation='relu', input_shape=(10,)),
    Dense(64, activation='relu'),
    Dense(2)
])

# Compile the model
model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))

# Example usage
print(model.summary())


########## GPT ######################################################################################################
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Load pre-trained GPT-2 model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

# Example usage
text = "Once upon a time"
input_ids = tokenizer.encode(text, return_tensors='pt')
output = model.generate(input_ids, max_length=50, num_return_sequences=1)
decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)
print(decoded_output)


########## BERT ######################################################################################################
from transformers import BertTokenizer, BertForQuestionAnswering
import torch

# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForQuestionAnswering.from_pretrained('bert-base-uncased')

# Example usage
question = "What is the capital of France?"
context = "The capital of France is Paris."
inputs = tokenizer.encode_plus(question, context, return_tensors="pt", add_special_tokens=True)
input_ids = inputs["input_ids"].tolist()
outputs = model(**inputs)
start_scores = outputs.start_logits
end_scores = outputs.end_logits
answer_start = torch.argmax(start_scores)
answer_end = torch.argmax(end_scores) + 1
answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids[0][answer_start:answer_end]))
print("Answer:", answer)


########## Fine Tuning ######################################################################################################
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
import torch

# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased')

# Example fine-tuning
optimizer = AdamW(model.parameters(), lr=5e-5)
train_dataset = ...  # Your training dataset
train_loader = ...   # Your data loader
for epoch in range(num_epochs):
    for batch in train_loader:
        inputs = tokenizer(batch['text'], return_tensors='pt', padding=True, truncation=True)
        labels = torch.tensor(batch['labels'])
        outputs = model(**inputs, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()


########## Text Summarization ######################################################################################################
from transformers import pipeline

# Load pre-trained model for text summarization
summarizer = pipeline("summarization")

# Example text
text = """
Your text goes here.
"""

# Generate summary
summary = summarizer(text, max_length=100, min_length=30, do_sample=False)
print("Summary:", summary[0]['summary_text'])


########## Question Answering ######################################################################################################
from transformers import pipeline

# Load pre-trained model for question answering
qa_pipeline = pipeline("question-answering")

# Example text and question
context = "Your context goes here."
question = "Your question goes here?"

# Perform question answering
answer = qa_pipeline(question=question, context=context)
print("Answer:", answer['answer'])
