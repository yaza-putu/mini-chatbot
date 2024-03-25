from gensim import corpora, models, similarities
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import string
import re

# Sample responses
responses = [
    "Hello!",
    "hi, ada yang bisa saya bantu?",
    "senang bertemu dengan anda",
    "hari yang cerah",
    "Goodbye!",
    "BPJS Ketenagakerjaan adalah badan hukum yang dibentuk untuk menyelenggarakan program Jaminan Kecelakaan Kerja, Jaminan Hari Tua, Jaminan Pensiun, Jaminan Kematian, dan Jaminan Kehilangan Pekerjaan yang bertujuan untuk memberikan perlindungan menyeluruh kepada seluruh pekerja di Indonesia.",
    "BPJS Program JKN berdiri sejak tahun 2014. Sebelum memperhitungkan intervensi pemerintah, defisit JKN sebesar Rp1,9 triliun pada tahun 2014, 9,4 triliun pada tahun 2015, 6,7 triliun pada tahun 2016, 13,8 triliun pada tahun 2017, dan 19,4 triliun pada tahun 2014. 2018.",
    "Alamat BPJS Jl. Letjen Suprapto Kav. 20 No.14 Jakarta Pusat 10510, Telp. 021 421 2938",
]

# Preprocess responses
stop_words = set(stopwords.words('indonesian'))
factory = StemmerFactory()
stemmer = factory.create_stemmer()

def preprocess_text(text):
    # case folding
    text = text.lower()
    # remove number
    text = re.sub(r"\d+", "", text)
    # remove white space leading & trailing
    text = text.strip()
    # remove multiple whitespace into single space
    text = re.sub('\s+', ' ', text)
    # tokenize
    tokens = word_tokenize(text)
    # Remove stopwords and punctuation
    tokens = [token for token in tokens if token not in stop_words and token not in string.punctuation]


    output =  [stemmer.stem(token) for token in tokens]
    return output

processed_responses = [preprocess_text(response) for response in responses]

# Build dictionary and corpus
dictionary = corpora.Dictionary(processed_responses)
corpus = [dictionary.doc2bow(response) for response in processed_responses]

# Create similarity index
tfidf = models.TfidfModel(corpus)
index = similarities.MatrixSimilarity(tfidf[corpus])

def get_most_similar_response(query):
    # Preprocess query
    query_tokens = preprocess_text(query)
    # Convert query to bag-of-words representation
    query_bow = dictionary.doc2bow(query_tokens)
    # Transform query to TF-IDF space
    query_tfidf = tfidf[query_bow]
    # Get similarity scores
    sims = index[query_tfidf]
    # Sort similarity scores
    sims = sorted(enumerate(sims), key=lambda item: -item[1])
    # Return the most similar response
    if sims[0][1] == 0.0:
        return "Sorry, i don't have knowledge about you ask"

    return responses[sims[0][0]]

# Chat loop
print("Bot: Hello! ada yang bisa saya bantu?")
while True:
    user_input = input("You: ")
    if user_input.lower() == 'exit':
        print("Bot: Goodbye!")
        break
    else:
        response = get_most_similar_response(user_input)
        print("Bot:", response)
