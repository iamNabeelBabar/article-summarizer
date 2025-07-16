import streamlit as st
import requests
from bs4 import BeautifulSoup
import nltk
import re
from nltk.tokenize import word_tokenize, sent_tokenize
from transformers import BartForConditionalGeneration, BartTokenizer
import torch
import os

# Ensure NLTK punkt resource is available
@st.cache_resource
def ensure_nltk_punkt():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        # Specify a writable directory for deployment
        nltk.download('punkt_tab', download_dir='/tmp/nltk_data')
        os.environ['NLTK_DATA'] = '/tmp/nltk_data'
    return True

ensure_nltk_punkt()

# Streamlit page configuration
st.set_page_config(page_title="News Summarizer", page_icon="📰", layout="wide")

# Load BART model and tokenizer globally
@st.cache_resource
def load_bart_model():
    model_name = "facebook/bart-large-cnn"
    tokenizer = BartTokenizer.from_pretrained(model_name)
    model = BartForConditionalGeneration.from_pretrained(model_name)
    if torch.cuda.is_available():
        model = model.to("cuda")
    return tokenizer, model

tokenizer, model = load_bart_model()

# Extract article content from a URL
@st.cache_data
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

# Preprocess text (keep numbers, punctuation)
def preprocess_text(text):
    text = re.sub(r'[^\w\s\.,:;/-]', '', text)
    tokens = word_tokenize(text.lower())
    clean_text = ' '.join(tokens)
    return ' '.join(clean_text.split()[:512])

# Improve final summary quality
def post_process_summary(summary, target_length):
    sentences = sent_tokenize(summary)
    if not sentences:
        return summary[:target_length]

    key_details = [s for s in sentences if re.search(r'\b\d{4}\b|olympics|venue|schedule|date|global|impact|format', s, re.IGNORECASE)]
    current_words, selected = 0, []

    for s in key_details + sentences:
        if s not in selected:
            wc = len(s.split())
            if current_words + wc <= target_length + 10:
                selected.append(s)
                current_words += wc

    summary = ' '.join(selected)
    if len(summary.split()) > target_length:
        words = summary.split()
        summary = ' '.join(words[:target_length])
        last_dot = summary.rfind('.')
        if last_dot != -1:
            summary = summary[:last_dot + 1]
    elif len(summary.split()) < target_length * 0.9:
        summary = ' '.join(sentences)  # Fallback to full summary if too short

    # Polish text
    summary = re.sub(r'\s+', ' ', summary).strip()
    summary = re.sub(r'\bolympics\b', 'the Olympics', summary, flags=re.IGNORECASE)
    summary = re.sub(r'\bt tournaments\b', 'T20 tournaments', summary, flags=re.IGNORECASE)
    return summary[0].upper() + summary[1:] if summary else ""

# Generate summary using BART
@st.cache_data
def summarize_text(text, max_length, _tokenizer, _model):
    try:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Tokenize input
        status_text.text("Tokenizing input...")
        inputs = _tokenizer(text, max_length=512, return_tensors="pt", truncation=True)
        if torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
        progress_bar.progress(70)

        # Generate summary
        status_text.text("Processing with BART...")
        with torch.no_grad():
            summary_ids = _model.generate(
                inputs['input_ids'],
                max_length=max_length + 20,
                min_length=max(30, max_length // 2),
                length_penalty=0.7,
                num_beams=4,  # Optimized for speed
                early_stopping=True
            )
        progress_bar.progress(95)
        
        # Decode and post-process
        status_text.text("Finalizing output...")
        summary = _tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        summary = post_process_summary(summary, max_length)
        progress_bar.progress(100)
        status_text.text("Summarization complete!")
        return summary
    except Exception as e:
        return f"Error during summarization: {str(e)}"
    finally:
        progress_bar.empty()
        status_text.empty()

# Streamlit App
def main():
    st.title("📰 News Article Summarizer")
    st.write("Enter a news article URL and get a clear, concise summary.")
    if not torch.cuda.is_available():
        st.info("Note: Using a GPU can significantly speed up summarization.")

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

            summary = summarize_text(cleaned, summary_length, tokenizer, model)
            
            if "Error" in summary:
                st.error(summary)
            else:
                st.subheader(f"📝 Summary ({summary_length} words):")
                st.write(summary)
                st.success("Summarization Complete ✅")
            progress.empty()
            status.empty()

if __name__ == "__main__":
    main()
