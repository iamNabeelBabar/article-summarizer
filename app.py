import streamlit as st
import requests
from bs4 import BeautifulSoup
import nltk
import re
from nltk.tokenize import word_tokenize, sent_tokenize
from transformers import BartForConditionalGeneration, BartTokenizer
import torch
import time

# Download NLTK data
nltk.download('punkt', quiet=True)

# Streamlit page configuration
st.set_page_config(page_title="News Summarizer", page_icon="📰", layout="wide")

# Function to extract article text from a URL
def extract_article_text(url):
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, 'html.parser')
        paragraphs = soup.find_all('p')
        article_text = ' '.join([para.get_text() for para in paragraphs])

        article_text = ' '.join(article_text.split())
        if not article_text:
            return "Error: No content found in the article."
        return article_text
    except Exception as e:
        return f"Error fetching article: {str(e)}"

# ✅ Updated function to preprocess text (keeps numbers and punctuation)
def preprocess_text(text):
    # Keep letters, numbers, and punctuation
    text = re.sub(r'[^\w\s\.,:;/-]', '', text)

    # Tokenize
    tokens = word_tokenize(text.lower())

    # Rejoin and limit
    clean_text = ' '.join(tokens)
    max_length = 1024
    clean_text = ' '.join(clean_text.split()[:max_length])
    return clean_text

# Function to post-process summary for clarity and completeness
def post_process_summary(summary, target_length):
    sentences = sent_tokenize(summary)
    if not sentences:
        return summary[:target_length]

    key_details = []
    for sentence in sentences:
        if re.search(r'\b\d{4}\b|olympics|venue|schedule|date', sentence, re.IGNORECASE):
            key_details.append(sentence)

    current_words = 0
    selected_sentences = []
    for sentence in key_details:
        word_count = len(sentence.split())
        if current_words + word_count <= target_length + 5:
            selected_sentences.append(sentence)
            current_words += word_count

    for sentence in sentences:
        if sentence not in selected_sentences:
            word_count = len(sentence.split())
            if current_words + word_count <= target_length + 5:
                selected_sentences.append(sentence)
                current_words += word_count

    final_summary = ' '.join(selected_sentences)

    if len(final_summary.split()) > target_length:
        words = final_summary.split()
        final_summary = ' '.join(words[:target_length])
        last_period = final_summary.rfind('.')
        if last_period != -1:
            final_summary = final_summary[:last_period + 1]

    final_summary = re.sub(r'\s+', ' ', final_summary).strip()
    if final_summary:
        final_summary = final_summary[0].upper() + final_summary[1:]
    return final_summary if final_summary else sentences[0][:target_length]

# Function to generate summary using BART with progress simulation
def summarize_text(text, max_length, progress_bar, status_text):
    try:
        model_name = "facebook/bart-large-cnn"
        tokenizer = BartTokenizer.from_pretrained(model_name)
        model = BartForConditionalGeneration.from_pretrained(model_name)

        status_text.text("Generating summary: Tokenizing input...")
        inputs = tokenizer(text, max_length=1024, return_tensors="pt", truncation=True)
        progress_bar.progress(70)

        status_text.text("Generating summary: Processing with BART...")
        for i in range(70, 95, 5):
            time.sleep(0.5)
            progress_bar.progress(i)

        summary_ids = model.generate(
            inputs['input_ids'],
            max_length=max_length + 20,
            min_length=max(30, max_length // 2),
            length_penalty=0.7,
            num_beams=10,
            early_stopping=True
        )

        status_text.text("Generating summary: Finalizing output...")
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        summary = post_process_summary(summary, max_length)
        progress_bar.progress(100)
        status_text.text("Summarization complete!")
        return summary
    except Exception as e:
        return f"Error during summarization: {str(e)}"

# Streamlit app interface
def main():
    st.title("News Article Summarizer")
    st.write("Enter a news article URL and select your desired summary length for a clear, concise summary.")

    with st.container():
        col1, col2 = st.columns([3, 1])
        with col1:
            url = st.text_input("Article URL", placeholder="https://example.com/news-article", key="url_input")
        with col2:
            summary_length = st.slider("Summary Length (words)", min_value=50, max_value=300, value=150, step=10)

    if st.button("Summarize", key="summarize_button"):
        if not url:
            st.error("Please enter a valid URL.")
        else:
            progress_bar = st.progress(0)
            status_text = st.empty()

            status_text.text("Extracting article content...")
            article_text = extract_article_text(url)
            progress_bar.progress(33)

            if "Error" in article_text:
                st.error(article_text)
                progress_bar.empty()
                status_text.empty()
            else:
                status_text.text("Preprocessing text for summarization...")
                cleaned_text = preprocess_text(article_text)
                progress_bar.progress(66)

                summary = summarize_text(cleaned_text, summary_length, progress_bar, status_text)

                if "Error" in summary:
                    st.error(summary)
                else:
                    st.subheader(f"Summary ({summary_length} words)")
                    st.write(summary)
                    st.success("Summarization complete!")
                progress_bar.empty()
                status_text.empty()

# Run the app
if __name__ == "__main__":
    main()
