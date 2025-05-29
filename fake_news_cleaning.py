import streamlit as st
import requests
import pandas as pd
import re
import nltk
import csv
import os
import logging
from datetime import datetime
from typing import List, Tuple, Optional
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from deep_translator import GoogleTranslator
from transformers import pipeline
import torch

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NewsClassifierApp:
    """Main application class for news classification"""
    
    def __init__(self):
        self.setup_nltk()
        self.setup_models()
        self.setup_directories()
    
    def setup_nltk(self):
        """Download required NLTK data"""
        try:
            nltk.download('stopwords', quiet=True)
            self.stop_words = set(stopwords.words('english'))
            self.stemmer = PorterStemmer()
        except Exception as e:
            logger.error(f"Error setting up NLTK: {e}")
            self.stop_words = set()
            self.stemmer = PorterStemmer()
    
    def setup_models(self):
        """Initialize translation and classification models"""
        try:
            self.translator = GoogleTranslator(source='auto', target='en')
            # Use a more reliable model with explicit device configuration
            self.classifier = pipeline(
                "text-classification", 
                model="unitary/toxic-bert",
                device=-1,  # Force CPU usage
                return_all_scores=False
            )
            logger.info("Models initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing models: {e}")
            st.error("Failed to load AI models. Please refresh the page.")
            self.classifier = None
            self.translator = None
    
    def setup_directories(self):
        """Create necessary directories"""
        os.makedirs("data", exist_ok=True)
        
        # Initialize feedback CSV with headers if it doesn't exist
        feedback_file = "data/feedback.csv"
        if not os.path.exists(feedback_file):
            with open(feedback_file, "w", newline='', encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["timestamp", "original_text", "translated_text", "predicted_label", "confidence", "user_feedback"])
    
    def clean_text(self, text: str) -> str:
        """Clean and preprocess text"""
        if not text or pd.isna(text):
            return ""
        
        try:
            text = str(text).lower()
            text = re.sub(r'\d+', '', text)  # Remove numbers
            text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
            text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
            
            words = text.split()
            words = [word for word in words if word not in self.stop_words and len(word) > 2]
            words = [self.stemmer.stem(word) for word in words]
            
            return ' '.join(words)
        except Exception as e:
            logger.error(f"Error cleaning text: {e}")
            return str(text)
    
    def translate_text(self, text: str) -> str:
        """Translate text to English"""
        if not self.translator:
            return text
        
        try:
            # Check if text is already in English (basic check)
            if self.is_english(text):
                return text
            
            translated = self.translator.translate(text)
            return translated if translated else text
        except Exception as e:
            logger.error(f"Translation error: {e}")
            return text
    
    def is_english(self, text: str) -> bool:
        """Basic check if text is in English"""
        try:
            # Simple heuristic: if most characters are ASCII, assume English
            ascii_chars = sum(1 for c in text if ord(c) < 128)
            return ascii_chars / len(text) > 0.8
        except:
            return True
    
    def classify_news(self, text: str) -> Tuple[str, str, float, str]:
        """Classify news as real or fake"""
        if not self.classifier:
            return "Error: Model not available", "", 0.0, "ERROR"
        
        try:
            # Clean and prepare text
            cleaned_text = self.clean_text(text)
            if not cleaned_text:
                return "Error: Empty text after cleaning", "", 0.0, "ERROR"
            
            # Translate if needed
            translated_text = self.translate_text(text)
            
            # Classify the text
            result = self.classifier(translated_text[:512])  # Limit text length
            
            if isinstance(result, list):
                result = result[0]
            
            label = result.get('label', 'UNKNOWN')
            confidence = float(result.get('score', 0.0))
            
            # Map labels to our classification system
            is_real = self.map_label_to_real(label, confidence)
            display_label = "خبر حقيقي ✅" if is_real else "خبر كاذب ❌"
            
            return display_label, translated_text, confidence, label
            
        except Exception as e:
            logger.error(f"Classification error: {e}")
            return f"Error: {str(e)}", "", 0.0, "ERROR"
    
    def map_label_to_real(self, label: str, confidence: float) -> bool:
        """Map model labels to real/fake classification"""
        label = label.upper()
        
        # Different models use different labels
        fake_labels = ['TOXIC', 'NEGATIVE', 'FAKE', 'LABEL_0']
        real_labels = ['NON_TOXIC', 'POSITIVE', 'REAL', 'LABEL_1']
        
        if label in fake_labels:
            return False
        elif label in real_labels:
            return True
        else:
            # Default to real if confidence is high, fake if low
            return confidence > 0.7
    
    def get_latest_news(self, api_key: str, query: str = "news", language: str = "en", page_size: int = 5) -> List[str]:
        """Fetch latest news from NewsAPI"""
        try:
            url = (
                'https://newsapi.org/v2/everything?'
                f'q={query}&'
                f'language={language}&'
                f'pageSize={page_size}&'
                'sortBy=publishedAt&'
                f'apiKey={api_key}'
            )
            
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            if data['status'] == 'ok':
                articles = data['articles']
                news_items = []
                
                for article in articles:
                    title = article.get('title', '')
                    description = article.get('description', '')
                    content = f"{title}. {description}" if description else title
                    
                    if content and content.strip():
                        news_items.append(content.strip())
                
                return news_items
            else:
                error_msg = data.get('message', 'Unknown error')
                st.error(f"NewsAPI Error: {error_msg}")
                return []
                
        except requests.exceptions.RequestException as e:
            st.error(f"Network error fetching news: {str(e)}")
            return []
        except Exception as e:
            st.error(f"Error fetching news: {str(e)}")
            return []
    
    def save_feedback(self, original_text: str, translated_text: str, predicted_label: str, confidence: float, feedback: str):
        """Save user feedback to CSV file"""
        try:
            timestamp = datetime.now().isoformat()
            feedback_data = [
                timestamp,
                original_text[:500],  # Limit text length
                translated_text[:500],
                predicted_label,
                confidence,
                feedback
            ]
            
            with open("data/feedback.csv", "a", newline='', encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(feedback_data)
            
            logger.info(f"Feedback saved: {feedback}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving feedback: {e}")
            st.error("Failed to save feedback. Please try again.")
            return False
    
    def display_news_classification(self, news_items: List[str]):
        """Display classified news items with feedback options"""
        if not news_items:
            st.warning("No news articles found.")
            return
        
        st.subheader("🗞️ الأخبار المصنفة:")
        
        for i, news in enumerate(news_items, 1):
            with st.container():
                st.markdown(f"### خبر رقم {i}")
                st.markdown(f"**النص الأصلي:** {news}")
                
                # Classify the news
                label, translated, confidence, raw_label = self.classify_news(news)
                
                if translated and translated != news:
                    st.markdown(f"**الترجمة:** {translated}")
                
                # Display classification with color coding
                if "حقيقي" in label:
                    st.success(f"🔎 **النتيجة:** {label} (الثقة: {confidence:.1%})")
                else:
                    st.error(f"🔎 **النتيجة:** {label} (الثقة: {confidence:.1%})")
                
                # Feedback buttons
                col1, col2, col3 = st.columns([1, 1, 2])
                
                with col1:
                    if st.button(f"✅ صحيح", key=f"correct_{i}"):
                        if self.save_feedback(news, translated, raw_label, confidence, "correct"):
                            st.success("تم حفظ التقييم كصحيح!")
                
                with col2:
                    if st.button(f"❌ خاطئ", key=f"wrong_{i}"):
                        if self.save_feedback(news, translated, raw_label, confidence, "wrong"):
                            st.warning("تم حفظ التقييم كخاطئ!")
                
                st.divider()
    
    def run(self):
        """Main application runner"""
        # Page configuration
        st.set_page_config(
            page_title="NewsTruth AI",
            page_icon="📰",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Custom CSS for better styling
        st.markdown("""
        <style>
        .main-header {
            text-align: center;
            color: #1f77b4;
            padding: 1rem 0;
        }
        .news-container {
            border: 1px solid #ddd;
            border-radius: 10px;
            padding: 1rem;
            margin: 1rem 0;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Main title
        st.markdown('<h1 class="main-header">📰 NewsTruth AI – تصنيف الأخبار العربية والإنجليزية</h1>', unsafe_allow_html=True)
        
        # Sidebar
        with st.sidebar:
            st.markdown("## 👤 معلومات المطور")
            st.markdown("**Riad Karkoura**")
            st.markdown("صحفي تقني | مختص بالذكاء الاصطناعي والتحقق من الأخبار")
            st.markdown("[🔗 LinkedIn](https://www.linkedin.com/in/riad-karkoura-b9010b196)")
            
            st.divider()
            
            # Display feedback statistics
            if os.path.exists("data/feedback.csv"):
                try:
                    df = pd.read_csv("data/feedback.csv")
                    st.markdown("## 📊 إحصائيات التقييم")
                    st.metric("إجمالي التقييمات", len(df))
                    if len(df) > 0:
                        correct_count = len(df[df['user_feedback'] == 'correct'])
                        st.metric("التقييمات الصحيحة", correct_count)
                        st.metric("معدل الدقة", f"{correct_count/len(df)*100:.1f}%")
                except:
                    pass
        
        # Main content
        tab1, tab2 = st.tabs(["🔍 تحليل الأخبار المباشر", "✍️ تحليل نص يدوي"])
        
        with tab1:
            st.subheader("جلب وتصنيف الأخبار من NewsAPI")
            
            # API Key input
            api_key = st.text_input(
                "🔑 أدخل مفتاح NewsAPI الخاص بك:",
                type="password",
                help="يمكنك الحصول على مفتاح مجاني من newsapi.org"
            )
            
            if api_key:
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    query = st.text_input(
                        "🔍 اكتب كلمة بحث:",
                        value="Syria OR vaccine",
                        help="استخدم OR للبحث عن عدة كلمات"
                    )
                
                with col2:
                    page_size = st.selectbox("عدد الأخبار:", [3, 5, 10], index=1)
                
                if st.button("📡 جلب وتصنيف الأخبار", type="primary"):
                    with st.spinner("جاري جلب وتحليل الأخبار..."):
                        # Detect language
                        lang = "ar" if any('\u0600' <= c <= '\u06FF' for c in query) else "en"
                        
                        # Fetch news
                        news_items = self.get_latest_news(api_key, query=query, language=lang, page_size=page_size)
                        
                        # Display results
                        self.display_news_classification(news_items)
            else:
                st.info("يرجى إدخال مفتاح NewsAPI للمتابعة")
        
        with tab2:
            st.subheader("تصنيف نص خبر يدوي")
            
            user_input = st.text_area(
                "أدخل نص الخبر هنا:",
                height=150,
                placeholder="اكتب أو الصق نص الخبر المراد تحليله..."
            )
            
            if st.button("🔍 تحليل الخبر", type="primary"):
                if user_input.strip():
                    with st.spinner("جاري تحليل النص..."):
                        label, translated, confidence, raw_label = self.classify_news(user_input)
                        
                        # Display results
                        if translated and translated != user_input:
                            st.markdown(f"**🔄 الترجمة:** {translated}")
                        
                        if "حقيقي" in label:
                            st.success(f"**🔍 التصنيف:** {label} (الثقة: {confidence:.1%})")
                        else:
                            st.error(f"**🔍 التصنيف:** {label} (الثقة: {confidence:.1%})")
                        
                        # Feedback section
                        st.subheader("📝 تقييم النتيجة")
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            if st.button("✅ التصنيف صحيح"):
                                if self.save_feedback(user_input, translated, raw_label, confidence, "correct"):
                                    st.success("شكراً لك! تم حفظ تقييمك.")
                        
                        with col2:
                            if st.button("❌ التصنيف خاطئ"):
                                if self.save_feedback(user_input, translated, raw_label, confidence, "wrong"):
                                    st.warning("شكراً لك! سنعمل على تحسين النموذج.")
                else:
                    st.warning("يرجى إدخال نص الخبر أولاً.")

# Initialize and run the app
if __name__ == "__main__":
    app = NewsClassifierApp()
    app.run()
