# News Article Summarization Tool

A powerful Streamlit web application designed to summarize news articles into concise ~100-word summaries using the state-of-the-art BART transformer model from Hugging Face. Users can input a news article URL, and the app automatically extracts the text, preprocesses it with natural language processing (NLP) techniques, and generates an abstractive summary. This tool is ideal for media companies, content creators, journalists, and anyone needing quick, high-quality summaries for newsletters, social media, or content curation.

## Table of Contents
1. [Project Overview](#project-overview)
2. [Features](#features)
3. [Technical Architecture](#technical-architecture)
4. [Requirements](#requirements)
5. [Setup Instructions](#setup-instructions)
6. [Usage](#usage)
7. [Example](#example)
8. [Dependencies](#dependencies)
9. [Testing and Validation](#testing-and-validation)
10. [Deployment Options](#deployment-options)
11. [Performance Optimization](#performance-optimization)
12. [Troubleshooting](#troubleshooting)
13. [Future Enhancements](#future-enhancements)
14. [License](#license)
15. [Contact](#contact)

## Project Overview
The **News Article Summarization Tool** leverages advanced NLP to streamline content consumption. It addresses the need for rapid summarization in the fast-paced media landscape by automating the process of extracting key insights from news articles. The app combines web scraping, NLP preprocessing, and transformer-based summarization to deliver concise, human-readable summaries. Built with Streamlit, it offers an intuitive interface suitable for non-technical users, making it a valuable tool for Upwork clients in media, marketing, or content creation.

### Use Cases
- **Media Companies**: Generate summaries for newsletters or breaking news updates.
- **Content Creators**: Create concise social media posts from lengthy articles.
- **Researchers**: Quickly grasp key points from news sources for analysis.
- **Business Professionals**: Summarize industry news for reports or briefings.

### Why This Tool?
- **Efficiency**: Reduces hours of manual summarization to seconds.
- **Accuracy**: Uses the `facebook/bart-large-cnn` model, fine-tuned for news summarization.
- **Accessibility**: Web-based interface requires no coding knowledge to use.
- **Scalability**: Easily adaptable for batch processing or API integration.

## Features
- **URL Input**: Users input a news article URL via a simple text field.
- **Web Scraping**: Extracts article text from HTML using robust parsing techniques.
- **NLP Preprocessing**: Cleans text by removing special characters, tokenizing, and truncating to fit model requirements.
- **BART Summarization**: Generates abstractive summaries (~100 words) that paraphrase key points for readability.
- **User-Friendly Interface**: Streamlit-based UI with progress indicators and error handling.
- **Error Feedback**: Displays clear error messages for invalid URLs or processing issues.
- **Customizable**: Easily extendable for adjustable summary lengths or multiple URLs.

## Technical Architecture
The tool follows a modular pipeline:
1. **Input Layer**: Streamlit collects the URL from the user.
2. **Web Scraping**: `requests` fetches the webpage, and `BeautifulSoup` extracts text from `<p>` tags.
3. **NLP Preprocessing**: `nltk` tokenizes and cleans text, removing noise (e.g., special characters) and truncating to 1024 tokens.
4. **Summarization**: The BART model (`facebook/bart-large-cnn`) generates an abstractive summary using transformer-based NLP.
5. **Output Layer**: Streamlit displays the summary or error messages with a responsive UI.

### Key Technologies
- **Streamlit**: Web framework for a user-friendly interface.
- **BeautifulSoup**: HTML parsing for text extraction.
- **NLTK**: NLP preprocessing (tokenization, cleaning).
- **Hugging Face Transformers**: BART model for abstractive summarization.
- **PyTorch**: Backend for running the BART model.
- **Requests**: HTTP requests for fetching article content.

## Requirements
- **Operating System**: Windows, macOS, or Linux.
- **Python**: Version 3.8 or higher.
- **Rust**: Required for compiling `tokenizers` (part of `transformers`). Install via [https://rustup.rs/](https://rustup.rs/).
- **Virtual Environment**: Recommended to isolate dependencies.
- **Dependencies**: Listed in `requirements.txt` (see below).
- **Hardware**: At least 8GB RAM; GPU recommended for faster summarization.

## Setup Instructions
Follow these steps to set up and run the application locally:

1. **Clone the Repository** (or download the project files):
   ```bash
   git clone <repository-url>
   cd news-article-summarizer
   ```
   Replace `<repository-url>` with the actual repository URL or copy files manually.

2. **Create and Activate a Virtual Environment**:
   ```bash
   python -m venv summarizer_env
   source summarizer_env/bin/activate  # On Windows: summarizer_env\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   Ensure Rust is installed (`rustc --version`) to avoid compilation errors for `tokenizers`.

4. **Run the Streamlit App**:
   ```bash
   streamlit run app.py
   ```

5. **Access the App**:
   - Open the URL provided in the terminal (e.g., `http://localhost:8501`) in a web browser.

### Prerequisites Installation
- **Python**: Download from [python.org](https://www.python.org/downloads/) (version 3.8+).
- **Rust**: Install via [rustup.rs](https://rustup.rs/). Verify with:
  ```bash
  rustc --version
  cargo --version
  ```
- **Dependencies**: Ensure `requirements.txt` is in the project directory.

## Usage
1. Launch the app:
   ```bash
   streamlit run app.py
   ```
2. Open the web interface in your browser (e.g., `http://localhost:8501`).
3. Enter a news article URL (e.g., from BBC, CNN, Reuters) in the input field.
4. Click the "Summarize" button.
5. View the generated ~100-word summary or any error messages (e.g., invalid URL or no content found).
6. Repeat with different URLs as needed.

### Tips
- Use URLs from reputable news sources for best results.
- Ensure the URL points to a single article, not a homepage or archive.
- Check for error messages to troubleshoot issues like blocked scraping.

## Example
- **Input URL**: `https://www.bbc.com/news/technology-69246886`
- **Output Example**:
  > A breakthrough in AI technology has been announced, with researchers developing a new model that enhances natural language understanding. The model, trained on diverse datasets, outperforms previous systems in tasks like summarization and question answering. Experts predict this could revolutionize applications in education and customer service. However, concerns about ethical use and data privacy remain, prompting calls for stricter regulations.
- **Word Count**: ~100 words, capturing the article’s key points in a concise, readable format.

## Dependencies
Listed in `requirements.txt`:
- `streamlit==1.39.0`: Web framework for the UI.
- `requests==2.32.3`: Fetches webpage content via HTTP.
- `beautifulsoup4==4.12.3`: Parses HTML for text extraction.
- `nltk==3.9.1`: Handles NLP preprocessing (tokenization, cleaning).
- `transformers==4.44.2`: Provides the BART model and tokenizer.
- `torch==2.4.1`: PyTorch backend for running BART.
- `tokenizers==0.19.1`: Tokenization for the BART model, requires Rust.

To install:
```bash
pip install -r requirements.txt
```

## Testing and Validation
The app has been tested with various news sources to ensure reliability:
- **Tested URLs**:
  - `https://www.bbc.com/news/technology-69246886`
  - `https://www.cnn.com/2025/07/16/tech/ai-breakthrough`
  - `https://www.reuters.com/technology/2025/07/15/innovation`
- **Test Scenarios**:
  - Valid URLs with standard HTML (`<p>` tags).
  - Invalid URLs (e.g., 404 errors).
  - Articles with minimal text (<50 words).
  - Sites with dynamic content (partial success, may require Selenium).
- **Validation**:
  - Summaries are coherent, ~100 words, and capture key points.
  - Error messages are clear for invalid inputs or scraping failures.

### How to Test
1. Run `streamlit run app.py`.
2. Test with at least 3-5 news article URLs from different sources.
3. Verify summary length and quality.
4. Check error handling for invalid URLs or blocked sites.

## Deployment Options
For production or client use, deploy the app to make it accessible online:
1. **Streamlit Cloud**:
   - Push the project to a GitHub repository.
   - Sign up at [streamlit.io/cloud](https://streamlit.io/cloud).
   - Connect your repository and deploy.
   - Share the generated URL with clients.
2. **Heroku**:
   - Create a `Procfile` with: `web: streamlit run app.py --server.port $PORT`.
   - Deploy using Heroku CLI: `heroku create`, `git push heroku main`.
3. **AWS EC2**:
   - Set up an EC2 instance with Python and Rust.
   - Clone the repository, install dependencies, and run `streamlit run app.py --server.port 80`.
4. **Local Deployment**:
   - Share `app.py`, `requirements.txt`, and `README.md` with clients for local use.

### Deployment Notes
- **Streamlit Cloud** is the easiest for quick demos.
- **Heroku/AWS** offer more control for production environments.
- **Client Setup**: Provide detailed instructions (as in this README) for local deployment.

## Performance Optimization
- **Model Choice**: The `facebook/bart-large-cnn` model is accurate but resource-intensive. For faster inference, switch to `distilbart-cnn-12-6`:
  ```python
  model_name = "distilbart-cnn-12-6"
  ```
- **Hardware**: Use a GPU-enabled system to speed up BART inference (requires `torch` with CUDA support).
- **Caching**: Add Streamlit caching to avoid reloading the model:
  ```python
  @st.cache_resource
  def load_model():
      model_name = "facebook/bart-large-cnn"
      tokenizer = BartTokenizer.from_pretrained(model_name)
      model = BartForConditionalGeneration.from_pretrained(model_name)
      return tokenizer, model
  ```
- **Text Truncation**: Already implemented to limit input to 1024 tokens, but further optimization can reduce processing time for long articles.

## Troubleshooting
- **Error: "No content found"**:
  - **Cause**: The article lacks `<p>` tags or uses dynamic content (JavaScript-rendered).
  - **Solution**: Modify `extract_article_text` to target other tags (e.g., `<div class="article-body">`) or use Selenium for dynamic sites.
- **Error: "Connection issues"**:
  - **Cause**: Invalid URL, server blocks, or network issues.
  - **Solution**: Verify the URL, ensure internet connectivity, or add headers:
    ```python
    headers = {'User-Agent': 'Mozilla/5.0'}
    ```
- **Error: "Rust not found"**:
  - **Cause**: Rust toolchain missing or not in PATH.
  - **Solution**: Reinstall Rust via [rustup.rs](https://rustup.rs/) and verify with `rustc --version`.
- **Slow Summarization**:
  - **Cause**: Large model or limited hardware.
  - **Solution**: Use `distilbart-cnn-12-6` or run on a GPU.
- **Installation Errors**:
  - **Cause**: Incompatible `tokenizers` version or missing Rust.
  - **Solution**: Use `requirements.txt` with `tokenizers==0.19.1` and ensure Rust is installed.

## Future Enhancements
- **Adjustable Summary Length**: Add a slider in Streamlit to let users choose summary length (e.g., 50-200 words).
- **Batch Processing**: Support multiple URLs for bulk summarization.
- **API Integration**: Replace web scraping with NewsAPI or similar for reliable text extraction.
- **Multilingual Support**: Use multilingual BART models (e.g., `facebook/mbart-large-50`) for non-English articles.
- **Export Options**: Allow users to download summaries as text or PDF.
- **Authentication**: Add user login for enterprise use.
- **Analytics Dashboard**: Track summarized articles and display metrics (e.g., word count, processing time).

## License
MIT License. You are free to use, modify, and distribute this software for personal or commercial purposes. See [LICENSE](LICENSE) for details.

## Contact
For support, customizations, or deployment assistance, contact the developer:
- **Upwork**: [Your Upwork Profile URL]
- **Email**: [Your Email Address]
- **GitHub**: [Your GitHub Profile URL]

### Support Notes
- **Response Time**: Within 24 hours via Upwork or email.
- **Customizations**: Available for additional features, deployment, or integration.
- **Maintenance**: Contact for updates to dependencies or model enhancements.