import json
import re
import nltk
import streamlit as st
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


nltk.download("stopwords", quiet=True)
stop_words = set(stopwords.words("english"))

with open("faqs.json", "r", encoding="utf-8") as file:
    faqs = json.load(file)


def preprocess(text):
    text = text.lower()
    words = re.findall(r"\b[a-z0-9]+\b", text)
    return " ".join(
        word for word in words if word not in stop_words
    )

questions = [faq["question"] for faq in faqs]
processed_questions = [preprocess(q) for q in questions]


vectorizer = TfidfVectorizer()
faq_vectors = vectorizer.fit_transform(processed_questions)


def get_answer(user_question):
    cleaned = preprocess(user_question)

    if not cleaned:
        return "Please enter a valid question."

    user_vector = vectorizer.transform([cleaned])
    scores = cosine_similarity(user_vector, faq_vectors)[0]
    best_match = scores.argmax()

    if scores[best_match] < 0.15:
        return "Sorry, I couldn't find an answer. Please try another question."

    return faqs[best_match]["answer"]


st.set_page_config(page_title="FAQ Chatbot", page_icon="💬")
st.title("💬 College FAQ Chatbot")
st.write("Ask me anything about the college!")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

user_input = st.chat_input("Type your question here...")

if user_input:
    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })

    answer = get_answer(user_input)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })

    st.rerun()