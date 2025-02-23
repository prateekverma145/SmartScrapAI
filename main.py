import streamlit as st
import requests
from newspaper import Article, Config
from urllib.parse import quote
from typing import List, Dict
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from string import punctuation
from heapq import nlargest
import time
import json
from duckduckgo_search import DDGS
from datetime import datetime
from deep_translator import GoogleTranslator
from deep_translator.exceptions import RequestError
import re
import unicodedata
import base64

# ------------------- Helper Function -------------------
def safe_translate(text, target_language, chunk_size=4900, max_retries=3):
    """
    Translate text in chunks to avoid deep_translator length limits.
    Retries translation up to max_retries times. On failure, returns the original chunk.
    """
    translator = GoogleTranslator(source='auto', target=target_language)
    translated_text = ""
    for i in range(0, len(text), chunk_size):
        chunk = text[i:i+chunk_size]
        for attempt in range(max_retries):
            try:
                translated_text += translator.translate(chunk)
                break  # Break out of retry loop on success
            except RequestError as e:
                if attempt == max_retries - 1:
                    # st.error(f"Translation failed for a chunk after {max_retries} attempts: {e}")
                    translated_text += chunk  # Fallback: append original text
                else:
                    time.sleep(1)  # Wait before retrying
    return translated_text

try:
    nltk.download('punkt_tab', quiet=True)
except:
    st.warning("Could not download punkt_tab. Attempting to download all NLTK data.")
    nltk.download('all', quiet=True)

# ------------------- NewsSearcher -------------------
class NewsSearcher:
    def __init__(self):
        self.config = Config()
        self.config.browser_user_agent = (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/91.0.4472.124 Safari/537.36"
        )
        self.search_settings = {
            'region': 'in-en',
            'safesearch': 'off',
            'timelimit': 'm',
            'max_results': 3
        }

    def search_news(self, query: str, location: str = None) -> List[Dict]:
        articles = []
        try:
             # Append "-site:msn.com -site:usnews.com" to exclude MSN and usnews.com from search results
            keywords = f"{query} {location} news -site:msn.com -site:usnews.com" if location else f"{query} news -site:msn.com -site:usnews.com"
            keywords = keywords.strip().replace("  ", " ")

            with DDGS() as ddgs:
                results = list(ddgs.news(
                    keywords=keywords,
                    region=self.search_settings['region'],
                    safesearch=self.search_settings['safesearch'],
                    timelimit=self.search_settings['timelimit'],
                    max_results=self.search_settings['max_results']
                ))

                for result in results:
                    article = {
                        'url': result['url'],
                        'source': result['source'],
                        'title': result['title'],
                        'text': result['body'],
                        'publish_date': result['date'],
                        'image_url': result.get('image', None)
                    }
                    articles.append(article)

        except Exception as e:
            st.error(f"Error in DuckDuckGo news search: {str(e)}")

        return articles
    

# ------------------- NewsProcessor -------------------
class NewsProcessor:
    def __init__(self):
        try:
            nltk.download(['punkt', 'stopwords', 'averaged_perceptron_tagger'], quiet=True)
            self.stopwords = set(stopwords.words('english') + list(punctuation))
        except Exception:
            self.stopwords = set(list(punctuation))

    def fetch_article(self, url: str) -> Dict:
        try:
            # Create a config with a custom browser user agent
            config = Config()
            config.browser_user_agent = (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/91.0.4472.124 Safari/537.36"
            )
            article = Article(url, config=config)
            article.download()
            time.sleep(1)
            article.parse()
            text = article.text.replace('\n', ' ').replace('\r', '')
            return {
                'title': article.title,
                'text': text,
                'url': url,
                'publish_date': article.publish_date,
                'image_url': article.top_image
            }
        except Exception:
            # Silently log error but return a valid dict with available information
            return {
                'title': "Article Preview Unavailable",
                'text': "Full article content could not be retrieved. You can visit the original source for complete information.",
                'url': url,
                'publish_date': None,
                'image_url': None
            }

    def summarize_text(self, text: str, num_sentences: int = 5) -> str:
        if not text:
            return ""
        try:
            sentences = text.split('. ')
            if len(sentences) <= num_sentences:
                return text
            words = text.lower().split()
            word_freq = {}
            for word in words:
                if word not in self.stopwords:
                    word_freq[word] = word_freq.get(word, 0) + 1
            sentence_scores = {}
            for sentence in sentences:
                for word in sentence.lower().split():
                    if word in word_freq:
                        sentence_scores[sentence] = sentence_scores.get(sentence, 0) + word_freq[word]
            summary_sentences = nlargest(num_sentences, sentence_scores, key=sentence_scores.get)
            return '. '.join(summary_sentences) + '.'
        except Exception as e:
            st.error(f"Error in summarization: {str(e)}")
            return text[:500] + "..."

# ------------------- HashnodePublisher -------------------
class HashnodePublisher:
    def __init__(self):
        self.api_token = "7d406b94-4b5b-4d53-8814-5a6a957a9564"
        self.publication_id = "67bb4bc06a1a10a27a4c1c07"
        self.api_url = "https://gql.hashnode.com/"
        self.headers = {
            'Authorization': self.api_token,
            'Content-Type': 'application/json'
        }
        try:
            nltk.download(['punkt', 'stopwords'], quiet=True)
        except:
            pass

    def _create_post_mutation(self) -> str:
        return """
        mutation PublishPost($input: PublishPostInput!) {
            publishPost(input: $input) {
                post {
                    id
                    title
                    slug
                    url
                }
            }
        }
        """

    def _slugify(self, text: str) -> str:
        text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('ascii')
        text = text.lower().strip()
        slug = re.sub(r'[^a-z0-9]+', '-', text)
        slug = slug.strip('-')
        return slug[:250]

    def _summarize_text(self, text: str, num_sentences: int = 5) -> str:
        from nltk.tokenize import sent_tokenize, word_tokenize
        from nltk.corpus import stopwords
        if not text:
            return ""
        stop_words = set(stopwords.words('english') + list(punctuation))
        sentences = sent_tokenize(text)
        if len(sentences) <= num_sentences:
            return text
        freq = {}
        for word in word_tokenize(text.lower()):
            if word not in stop_words:
                freq[word] = freq.get(word, 0) + 1
        sentence_scores = {}
        for sentence in sentences:
            for word in word_tokenize(sentence.lower()):
                if word in freq:
                    sentence_scores[sentence] = sentence_scores.get(sentence, 0) + freq[word]
        top_sentences = nlargest(num_sentences, sentence_scores, key=sentence_scores.get)
        summary = " ".join(top_sentences)
        return summary

    def generate_image(self, article: dict) -> str:
        try:
            prompt = article.get('title', '')
            summary = article.get('summary', '')
            if summary:
                prompt += f" - {summary[:100]}"
            encoded_prompt = quote(prompt, safe='')
            image_url = f"https://image.pollinations.ai/prompt/{encoded_prompt}"
            response = requests.head(image_url)
            if response.status_code == 200:
                return image_url
            else:
                # st.warning(f"Could not generate AI image for: {article.get('title', 'Untitled')}")
                return None
        except Exception as e:
            # st.warning(f"Error generating AI image: {str(e)}")
            return None

    def publish_combined_article(self, articles, topic: str, location: str = None, language: str = "en") -> dict:
        for article in articles:
            ai_image = self.generate_image(article)
            if ai_image:
                article['ai_image_url'] = ai_image

        # Create the original title in English for slug generation.
        original_title = f"News Roundup: {topic.title()}"
        if location:
            original_title += f" in {location.title()}"
        
        # Generate slug from the original title.
        slug = self._slugify(original_title)
        if not slug:
            slug = f"news-roundup-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        # Translate title only for display if needed.
        if language != "en":
            display_title = safe_translate(original_title, language)
        else:
            display_title = original_title

        # Format combined content (the translation of the content is handled inside the method)
        content = self.format_combined_content(articles, topic, location, language)
        
        cover_image = None
        if articles and articles[0].get('image_url'):
            cover_image_url = articles[0]['image_url'].rstrip("\\/")
            cover_image = {"coverImageURL": cover_image_url}
        
        variables = {
            "input": {
                "title": display_title,
                "contentMarkdown": content,
                "slug": slug,
                "publicationId": self.publication_id,
                "tags": [
                    {"name": "News", "slug": "news"}
                ],
                "disableComments": False,
                "coverImageOptions": cover_image
            }
        }
        try:
            response = requests.post(
                self.api_url,
                headers=self.headers,
                json={'query': self._create_post_mutation(), 'variables': variables}
            )
            if response.status_code == 200:
                result = response.json()
                if 'errors' in result:
                    st.error(f"Hashnode API Error:\n{json.dumps(result['errors'], indent=2)}")
                    return None
                return result.get('data', {}).get('publishPost', {}).get('post')
            else:
                st.error(f"HTTP Error: {response.status_code}\nResponse: {response.text}")
                return None
        except Exception as e:
            st.error(f"Error publishing article: {str(e)}")
            return None


    def format_combined_content(self, articles, topic: str, location: str = None, language: str = "en") -> str:
        current_date = datetime.now().strftime("%Y-%m-%d")
        combined_text = ""
        for article in articles:
            if article.get('text'):
                combined_text += article['text'] + " "
            elif article.get('summary'):
                combined_text += article['summary'] + " "
        combined_summary = self._summarize_text(combined_text, num_sentences=5)
        content = f"# News Roundup: {topic.title()}"
        if location:
            content += f" in {location.title()}"
        content += f"\n\n*Published on {current_date}*\n\n"
        content += "## Introduction\n"
        content += f"Below you'll find a curated overview of the latest news about **{topic}**"
        if location:
            content += f" in **{location}**"
        content += ". This post aggregates multiple sources and includes both original and AI-generated images.\n\n"
        content += "## Combined Summary\n"
        content += combined_summary + "\n\n"
        content += "## Detailed Summaries\n\n"
        for idx, article in enumerate(articles, 1):
            title = article.get('title', '').strip() or f"Article #{idx}"
            content += f"### {idx}. {title}\n\n"
            source_name = article.get('source', 'Unknown Source')
            source_url = article.get('url', '')
            content += f"**Source**: {source_name}\n\n"
            if source_url:
                content += f"**Read Full Article**: [Link]({source_url})\n\n"
            per_article_summary = article.get('summary', '')
            if per_article_summary:
                content += f"**Article Summary**:\n\n{per_article_summary}\n\n"
            if article.get('image_url'):
                content += "**Original Image**:\n\n"
                content += f"![Original Article Image]({article['image_url']})\n\n"
            if article.get('ai_image_url'):
                content += "**AI-Generated Illustration**:\n\n"
                content += f"![AI Generated Illustration]({article['ai_image_url']})\n\n"
                content += "*AI-generated image related to this article.*\n\n"
            content += "---\n\n"
        content += "\n\n---\n"
        content += "*This news roundup was automatically curated and published using AI. "
        content += f"Last updated: {current_date}*"
        if language != "en":
            content = safe_translate(content, language)
        return content

# ------------------- Streamlit App -------------------
def main():
    # Set page config
    st.set_page_config(
        page_title="QuickNews – Fast, Reliable, Personalized",
        page_icon="📡",
        layout="wide"
    )

    st.markdown("""
        <style>
        .article-headline {
            font-size: 24px !important;
            font-weight: bold !important;
            color: #ffffff !important;
            margin-bottom: 1rem !important;
        }
        .article-description {
            font-size: 16px !important;
            color: #c0c0c0 !important;
            margin: 15px 0 !important;
            line-height: 1.6 !important;
            padding: 10px !important;
            background-color: rgba(255, 255, 255, 0.05) !important;
            border-radius: 5px !important;
        }
        .article-metadata {
            font-size: 14px !important;
            color: #8b949e !important;
            margin-top: 10px !important;
        }
        .source-tag {
            background-color: #1e3a8a !important;
            padding: 2px 8px !important;
            border-radius: 4px !important;
            font-size: 12px !important;
        }

        </style>
    """, unsafe_allow_html=True)

    # Initialize session state
    if "processed_articles" not in st.session_state:
        st.session_state.processed_articles = []
    if "search_query" not in st.session_state:
        st.session_state.search_query = ""
    if "location" not in st.session_state:
        st.session_state.location = ""
    if "language" not in st.session_state:
        st.session_state.language = "en"

    # Header section with logo and title
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.markdown("# 🕵️")
    
    with col2:
        st.title("QuickNews – Fast, Reliable, Personalized")

    # Display Image
    st.image("./assets/finalimg.jpg", use_container_width =True)  
    

    st.markdown("---")

    # Language mapping
    language_map = {
        "en": "English", "es": "Spanish", "fr": "French",
        "de": "German", "it": "Italian", "pt": "Portuguese",
        "hi": "Hindi", "ta": "Tamil", "te": "Telugu",
        "ml": "Malayalam", "bn": "Bengali"
    }
    language_names = list(language_map.values())

    # Search section
    st.markdown("### 🔍 Search Parameters")
    col1, col2 = st.columns(2)
    
    with col1:
        search_query = st.text_input(
            "News Topic",
            value=st.session_state.search_query,
            placeholder="Enter a topic to search..."
        )
        
    with col2:
        location = st.text_input(
            "Location (Optional)",
            value=st.session_state.location,
            placeholder="Enter a location..."
        )

    col3, col4 = st.columns(2)
    
    with col3:
        default_language_name = language_map.get(st.session_state.language, "English")
        selected_language_name = st.selectbox(
            "Display Language",
            options=language_names,
            index=language_names.index(default_language_name)
        )
        selected_language_code = [code for code, name in language_map.items() if name == selected_language_name][0]
        st.session_state.language = selected_language_code

    with col4:
        st.markdown("<br>", unsafe_allow_html=True)
        search_button = st.button("🔎 Search News")



    if search_button:
        st.session_state.search_query = search_query
        st.session_state.location = location
        if search_query:
            with st.spinner("🔄 Searching and processing news articles..."):
                try:
                    searcher = NewsSearcher()
                    processor = NewsProcessor()
                    articles_info = searcher.search_news(search_query, location)
                    
                    if articles_info:
                        results_container = st.container()
                        with results_container:
                            st.markdown("### 📚 Search Results")
                            
                            seen_titles = set()
                            unique_articles = []
                            processed_articles = []
                            
                            progress_bar = st.progress(0)
                            total_articles = len(articles_info)
                            
                            for idx, art in enumerate(articles_info):
                                if art['title'] not in seen_titles:
                                    seen_titles.add(art['title'])
                                    unique_articles.append(art)
                                    
                                    progress = (idx + 1) / total_articles
                                    progress_bar.progress(progress)
                                    
                                    # Enhanced article display with larger headline and description
                                    st.markdown(f"""
                                        <div class="article-headline">
                                            {art['title']}
                                        </div>
                                    """, unsafe_allow_html=True)
                                    
                                    col1, col2 = st.columns([1, 2])
                                    with col1:
                                        if art.get('image_url'):
                                            st.image(art['image_url'], use_container_width=True)
                                    with col2:
                                        # Add article description
                                        if art.get('body'):
                                            description = art['body'][:300] + "..." if len(art['body']) > 300 else art['body']
                                            st.markdown(f"""
                                                <div class="article-description">
                                                    {description}
                                                </div>
                                            """, unsafe_allow_html=True)
                                        
                                        # Metadata section
                                        st.markdown(f"""
                                            <div class="metadata">
                                                <strong>Source:</strong> {art['source']}<br>
                                                <strong>Published:</strong> {art.get('publish_date', 'Date not available')}<br>
                                            </div>
                                        """, unsafe_allow_html=True)
                                        
                                        st.markdown(f"**URL:** [{art['url']}]({art['url']})")
                                        
                                    # Process full article
                                    article_data = processor.fetch_article(art['url'])
                                    if article_data:
                                        if article_data.get('text'):
                                            article_data['summary'] = processor.summarize_text(article_data['text'])
                                        article_data['source'] = art['source']
                                        article_data['publish_date'] = (art['publish_date'] or article_data['publish_date'])
                                        processed_articles.append(article_data)
                                        
                                    st.markdown("---")
                            
                            progress_bar.empty()
                            
                            # Translation handling
                            if st.session_state.language != "en":
                                with st.spinner("🌐 Translating content..."):
                                    for idx, article in enumerate(processed_articles):
                                        for key in ['title', 'text', 'summary']:
                                            if article.get(key):
                                                article[key] = safe_translate(article[key], st.session_state.language)
                                        processed_articles[idx] = article
                            
                            st.session_state.processed_articles = processed_articles
                            
                            if processed_articles:
                                st.success(f"✅ Successfully processed {len(processed_articles)} articles")
                    else:
                        st.info("No articles found for your search criteria. Try different keywords.", icon="ℹ️")
                except Exception as e:
                    st.error("Unable to complete the search. Please try again.", icon="🚫")
        else:
            st.warning("Please enter a search topic.", icon="⚠️")

    # Publication section
    if st.session_state.processed_articles:
        st.markdown("---")
        st.markdown("### 📤 Publication")
        publish_col1, publish_col2 = st.columns([3, 1])
        
        with publish_col1:
            st.info(f"📝 Found {len(st.session_state.processed_articles)} articles ready for publication", icon="ℹ️")
            
        with publish_col2:
            if st.button("🚀 Publish to Hashnode"):
                with st.spinner("📡 Publishing to Hashnode..."):
                    publisher = HashnodePublisher()
                    result = publisher.publish_combined_article(
                        st.session_state.processed_articles,
                        st.session_state.search_query,
                        st.session_state.location,
                        st.session_state.language
                    )
                    if result:
                        st.success(f"✅ Published successfully! [View Article]({result['url']})", icon="✅")
                    else:
                        st.error("❌ Failed to publish article. Please try again.", icon="❌")

if __name__ == "__main__":
    main()
