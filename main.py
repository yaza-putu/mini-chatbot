from gensim import corpora, models, similarities
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string

# Sample responses
responses = [
    "Hello!",
    "How can I help you?",
    "Hi, nice to meet you",
    "Have a nice day!",
    "Goodbye!",
    "BPJS Employment is a legal entity formed to administer Work Accident Insurance, Old Age Security, Pension Security, Death Insurance and Job Loss Insurance programs which aim to provide complete protection to all workers in Indonesia.",
    "The BPJS stand The JKN program has been running behind schedule since 2014. Before accounting for government intervention, the JKN deficit was worth IDR 1.9 trillion in 2014, 9.4 trillion in 2015, 6.7 trillion in 2016, 13.8 trillion in 2017, and 19.4 trillion in 2018.",
    "The address of BPJS is Jl. Letjen Suprapto Kav. 20 No.14 Jakarta Pusat 10510, Telp. 021 421 2938",
]

# Preprocess responses
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    # Tokenize
    tokens = word_tokenize(text.lower())
    # Remove stopwords and punctuation
    tokens = [token for token in tokens if token not in stop_words and token not in string.punctuation]
    return tokens

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
print("Bot: Hello! How can I help you?")
while True:
    user_input = input("You: ")
    if user_input.lower() == 'exit':
        print("Bot: Goodbye!")
        break
    else:
        response = get_most_similar_response(user_input)
        print("Bot:", response)
