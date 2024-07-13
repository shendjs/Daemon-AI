import random
import nltk

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from textblob import TextBlob

import numpy as np

from Responses.general_responses import general_user_inputs, general_bot_responses
from Responses.info_responses import info_user_inputs, info_bot_responses
from Responses.history_responses import history_user_inputs, history_bot_responses
from Responses.software_responses import software_user_inputs, software_bot_responses

# NLTK data download (only needed on first run)
nltk.download('punkt')
nltk.download('stopwords')

# Natural language processing function
def process_text(input_text):
    input_text = input_text.lower()
    words = word_tokenize(input_text)

    stop_words = set(stopwords.words('english'))  # Load stop words list (use 'english' for English)
    filtered_words = [word for word in words if word not in stop_words]
    return ' '.join(filtered_words)

# Function for the bot to generate a response
def generate_response(user_input):
    # Process the user input
    processed_input = process_text(user_input)

    # TF-IDF vectorization
    all_inputs = general_user_inputs + info_user_inputs + history_user_inputs + software_user_inputs
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(all_inputs)

    # Vectorize the user input
    input_vector = tfidf_vectorizer.transform([processed_input])

    # Calculate similarity scores
    similarity_scores = cosine_similarity(input_vector, tfidf_matrix)

    # Find the index of the highest similarity score
    best_index = np.argmax(similarity_scores)

    # Determine the response
    if best_index < len(general_user_inputs):
        input_key = general_user_inputs[best_index]
        response = random.choice(general_bot_responses[input_key])
    elif best_index < len(general_user_inputs) + len(info_user_inputs):
        input_key = info_user_inputs[best_index - len(general_user_inputs)]
        response = random.choice(info_bot_responses[input_key])
    elif best_index < len(general_user_inputs) + len(info_user_inputs) + len(history_user_inputs):
        input_key = history_user_inputs[best_index - len(general_user_inputs) - len(info_user_inputs)]
        response = random.choice(history_bot_responses[input_key])
    else:
        input_key = software_user_inputs[best_index - len(general_user_inputs) - len(info_user_inputs) - len(history_user_inputs)]
        response = random.choice(software_bot_responses[input_key])

    return response

# User interface
def run_chatbot():
    print("-------------------------------------------------------------")
    print("Daemon: Hello! You can start chatting with me.")
    print("Daemon: Ask your questions or send a message.")
    print("Daemon: Press 'q' to quit.")
    print("-------------------------------------------------------------")
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'q':
            print("Daemon: Goodbye!")
            break
        
        response = generate_response(user_input)
        print("Daemon:", response)
        print("-------------------------------------------------------------")

if __name__ == "__main__":
    run_chatbot()
