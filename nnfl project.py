# -*- coding: utf-8 -*-
'''nnfl project - pre release movie ratings prediction'''

import pandas as pd
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Dense, Concatenate, GlobalAveragePooling1D, Flatten
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('tmdb_movies_data.csv')

df = df[['budget', 'cast', 'director', 'keywords', 'runtime', 'genres', 'production_companies', 'release_date', 'vote_average']]

df = df.dropna()
df['release_date'] = pd.to_datetime(df['release_date'])
df['release_month'] = df['release_date'].dt.month

def preprocess_text_column(column, max_len=5, num_words=10000):
    tokenizer = Tokenizer(num_words=num_words, split='|', lower=True)
    tokenizer.fit_on_texts(df[column].values)
    sequences = tokenizer.texts_to_sequences(df[column].values)
    padded_sequences = pad_sequences(sequences, maxlen=max_len)
    return padded_sequences, tokenizer

all_cast_members = df['cast'].str.split('|').explode()
unique_cast_members = set(all_cast_members)
num_unique_cast = len(unique_cast_members)

all_directors = df['director'].str.split('|').explode()
unique_directors = set(all_directors)
num_unique_directors = len(unique_directors)

all_genres = df['genres'].str.split('|').explode()
unique_genres = set(all_genres)
num_unique_genres = len(unique_genres)

all_companies = df['production_companies'].str.split('|').explode()
unique_companies = set(all_companies)
num_unique_companies = len(unique_companies)

print(f"Unique directors: {num_unique_directors}")
print(f"Unique cast members: {num_unique_cast}")
print(f"Unique genres: {num_unique_genres}")
print(f"Unique production companies: {num_unique_companies}")

cast_data, cast_tokenizer = preprocess_text_column('cast', max_len=5, num_words=num_unique_cast)
director_data, director_tokenizer = preprocess_text_column('director', max_len=1, num_words=num_unique_directors)
keywords_data, keywords_tokenizer = preprocess_text_column('keywords', max_len=5, num_words=5000)
genres_data, genres_tokenizer = preprocess_text_column('genres', max_len=3, num_words=num_unique_genres)
companies_data, companies_tokenizer = preprocess_text_column('production_companies', max_len=3, num_words=num_unique_companies)

numerical_data = df[['budget', 'runtime', 'release_month']].values
scaler = StandardScaler()
numerical_data = scaler.fit_transform(numerical_data)

# Target variable
y = df['vote_average'].values

# Train/test split
X_cast_train, X_cast_test, X_director_train, X_director_test, X_keywords_train, X_keywords_test, \
X_genres_train, X_genres_test, X_companies_train, X_companies_test, X_num_train, X_num_test, y_train, y_test = train_test_split(
    cast_data, director_data, keywords_data, genres_data, companies_data, numerical_data, y, test_size=0.2, random_state=42)

# Neural Network Model
input_cast = Input(shape=(5,), name="cast_input")
embedding_cast = Embedding(input_dim=num_unique_cast + 1, output_dim=32)(input_cast)
avg_cast_embeddings = GlobalAveragePooling1D()(embedding_cast)

input_director = Input(shape=(1,), name="director_input")
embedding_director = Embedding(input_dim=num_unique_directors + 1, output_dim=16)(input_director)
flatten_director = Flatten()(embedding_director)

input_keywords = Input(shape=(5,), name="keywords_input")
embedding_keywords = Embedding(input_dim=5000, output_dim=16)(input_keywords)
avg_keywords_embeddings = GlobalAveragePooling1D()(embedding_keywords)

input_genres = Input(shape=(3,), name="genres_input")
embedding_genres = Embedding(input_dim=num_unique_genres + 1, output_dim=8)(input_genres)
avg_genres_embeddings = GlobalAveragePooling1D()(embedding_genres)


input_companies = Input(shape=(3,), name="companies_input")
embedding_companies = Embedding(input_dim=num_unique_companies + 1, output_dim=16)(input_companies)
avg_companies_embeddings = GlobalAveragePooling1D()(embedding_companies)

#Numerical Input (budget, runtime, release_month)
input_numerical = Input(shape=(3,), name="numerical_input")
dense_numerical = Dense(16, activation='relu')(input_numerical)

# Combine all inputs
combined_features = Concatenate()([
    avg_cast_embeddings, flatten_director, avg_keywords_embeddings, 
    avg_genres_embeddings, avg_companies_embeddings, dense_numerical
])

# Fully connected layers
x = Dense(128, activation='relu')(combined_features)
x = Dense(64, activation='relu')(x)
x = Dense(32, activation='relu')(x)

# Output layer
output = Dense(1, name='output')(x)

model = Model(inputs=[input_cast, input_director, input_keywords, input_genres, input_companies, input_numerical], outputs=output)
model.compile(optimizer='adam', loss='mean_squared_error')

model.summary()

history = model.fit(
    x = [X_cast_train, X_director_train, X_keywords_train, X_genres_train, X_companies_train, X_num_train],
    y = y_train,
    validation_data=[[X_cast_test, X_director_test, X_keywords_test, X_genres_test, X_companies_test, X_num_test], y_test],
    epochs=15,
    batch_size=64
)