import streamlit as st
import requests
from bs4 import BeautifulSoup
import nltk
import re
from nltk.tokenize import word_tokenize, sent_tokenize
from transformers import BartForConditionalGeneration, BartTokenizer
import torch
import time

# ✅ Download NLTK resources safely
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt_tab')

# ✅ Streamlit page config
st.set_page_config(page_title="News Summarizer", page_icon="📰", layout="wide")

# ✅ Extract article content from a URL
def extract_article_text(url):
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        paragraphs = soup.find_all('p')
        article_text = ' '.join([p.get_text() for p in paragraphs])
        article_text = ' '.join(article_text.split())
        return article_text if article_text else "Error: No content found."
    except Exception as e:
        return f"Error fetching article: {str(e)}"

# ✅ Preprocess text (keep numbers, punctuation)
def preprocess_text(text):
    text = re.sub(r'[^\w\s\.,:;/-]', '', text)
    tokens = word_tokenize(text.lower())
    clean_text = ' '.join(tokens)
    return ' '.join(clean_text.split()[:1024])

# ✅ Improve final summary quality
def post_process_summary(summary, target_length):
    sentences = sent_tokenize(summary)
    if not sentences:
        return summary[:target_length]

    key_details = [s for s in sentences if re.search(r'\b\d{4}\b|olympics|venue|schedule|date', s, re.IGNORECASE)]
    current_words, selected = 0, []

    for s in key_details + sentences:
        if s not in selected:
            wc = len(s.split())
            if current_words + wc <= target_length + 5:
                selected.append(s)
                current_words += wc

    summary = ' '.join(selected)
    if len(summary.split()) > target_length:
        words = summary.split()
        summary = ' '.join(words[:target_length])
        last_dot = summary.rfind('.')
        if last_dot != -1:
            summary = summary[:last_dot + 1]
    return summary[0].upper() + summary[1:].strip() if summary else ""

# ✅ Generate summary using BART model
def summarize_text(text, max_length, progress_bar, status_text):
    try:
        model_name = "facebook/bart-large-cnn"
        tokenizer = BartTokenizer.from_pretrained(model_name)
        model = BartForConditionalGeneration.from_pretrained(model_name)

        status_text.text("Tokenizing input...")
        inputs = tokenizer(text, max_length=1024, return_tensors="pt", truncation=True)
        progress_bar.progress(70)

        status_text.text("Processing with BART...")
        for i in range(70, 95, 5):
            time.sleep(0.4)
            progress_bar.progress(i)

        with torch.no_grad():
            summary_ids = model.generate(
                inputs['input_ids'],
                max_length=max_length + 20,
                min_length=max(30, max_length // 2),
                length_penalty=0.7,
                num_beams=10,
                early_stopping=True
            )

        status_text.text("Finalizing output...")
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        progress_bar.progress(100)
        return post_process_summary(summary, max_length)
    except Exception as e:
        return f"Error during summarization: {str(e)}"

# ✅ Streamlit App
def main():
    st.title("📰 News Article Summarizer")
    st.write("Enter a news article URL and get a clear, concise summary.")

    with st.container():
        col1, col2 = st.columns([3, 1])
        with col1:
            url = st.text_input("Article URL", placeholder="https://example.com/news")
        with col2:
            summary_length = st.slider("Summary Length (words)", 50, 300, 150, 10)

    if st.button("Summarize"):
        if not url:
            st.error("Please enter a valid URL.")
        else:
            progress = st.progress(0)
            status = st.empty()

            status.text("Extracting article...")
            article = extract_article_text(url)
            progress.progress(33)

            if "Error" in article:
                st.error(article)
                progress.empty()
                status.empty()
                return

            status.text("Preprocessing text...")
            cleaned = preprocess_text(article)
            progress.progress(66)

            summary = summarize_text(cleaned, summary_length, progress, status)
            progress.empty()
            status.empty()

            if "Error" in summary:
                st.error(summary)
            else:
                st.subheader(f"📝 Summary ({summary_length} words):")
                st.write(summary)
                st.success("Summarization Complete ✅")

if __name__ == "__main__":
    main()
