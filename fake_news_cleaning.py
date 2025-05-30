import streamlit as st
import requests
import pandas as pd
import re
import nltk
import csv
import os
import logging
from datetime import datetime
from typing import List, Tuple
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from deep_translator import GoogleTranslator
from transformers import pipeline

# يجب استدعاء set_page_config أول شيء بعد الاستيرادات
st.set_page_config(
    page_title="NewsTruth AI",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
            self.classifier = pipeline(
                "text-classification",
                model="unitary/toxic-bert",
                device=-1,
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
            text = re.sub(r'\d+', '', text)
            text = re.sub(r'[^\w\s]', '', text)
            text = re.sub(r'\s+', ' ', text)

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
            ascii_chars = sum(1 for c in text if ord(c) < 128)
            return ascii_chars / len(text) > 0.8
        except Exception:
            return True

    def classify_news(self, text: str) -> Tuple[str, str, float, str]:
        """Classify news as real or fake"""
        if not self.classifier:
            return "Error: Model not available", "", 0.0, "ERROR"
        try:
            cleaned_text = self.clean_text(text)
            if not cleaned_text:
                return "Error: Empty text after cleaning", "", 0.0, "ERROR"
            translated_text = self.translate_text(text)
            result = self.classifier(translated_text[:512])
            if isinstance(result, list):
                result = result[0]
            label = result.get('label', 'UNKNOWN')
            confidence = float(result.get('score', 0.0))
            is_real = self.map_label_to_real(label, confidence)
            display_label = "خبر حقيقي ✅" if is_real else "خبر كاذب ❌"
            return display_label, translated_text, confidence, label
        except Exception as e:
            logger.error(f"Classification error: {e}")
            return f"Error: {str(e)}", "", 0.0, "ERROR"

    def map_label_to_real(self, label: str, confidence: float) -> bool:
        """Map model labels to real/fake classification"""
        label = label.upper()
        fake_labels = ['TOXIC', 'NEGATIVE', 'FAKE', 'LABEL_0']
        real_labels = ['NON_TOXIC', 'POSITIVE', 'REAL', 'LABEL_1']
        if label in fake_labels:
            return False
        elif label in real_labels:
            return True
        else:
            return confidence > 0.7

    def get_latest_news(self, api_key: str, query: str = "news", language: str = "en", page_size: int = 5) -> List[str]:
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
        try:
            timestamp = datetime.now().isoformat()
            feedback_data = [
                timestamp,
                original_text[:500],
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
        if not news_items:
            st.warning("No news articles found.")
            return
        st.subheader("🗞️ الأخبار المصنفة:")
        for i, news in enumerate(news_items, 1):
            with st.container():
                st.markdown(f"### خبر رقم {i}")
                st.markdown(f"**النص الأصلي:** {news}")
                label, translated, confidence, raw_label = self.classify_news(news)
                if translated and translated != news:
                    st.markdown(f"**الترجمة:** {translated}")
                if "حقيقي" in label:
                    st.success(f"🔎 **النتيجة:** {label} (الثقة: {confidence:.1%})")
                else:
                    st.error(f"🔎 **النتيجة:** {label} (الثقة: {confidence:.1%})")
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

    def classify_csv(self, df: pd.DataFrame, text_column: str) -> pd.DataFrame:
        """Classify news in CSV file and add results columns"""
        if text_column not in df.columns:
            st.error(f"عمود '{text_column}' غير موجود في الملف.")
            return df
        results = []
        for text in df[text_column]:
            label, translated, confidence, raw_label = self.classify_news(str(text))
            results.append({
                "original_text": text,
                "translated_text": translated,
                "predicted_label": label,
                "confidence": confidence,
                "raw_label": raw_label
            })
        results_df = pd.DataFrame(results)
        return pd.concat([df, results_df], axis=1)

    def run(self):
        st.title("📰 NewsTruth AI – تصنيف الأخبار الحقيقية والكاذبة")

        st.sidebar.header("الإعدادات")
        api_key = st.sidebar.text_input("أدخل مفتاح API الخاص بـ NewsAPI", type="password")
        query = st.sidebar.text_input("كلمة البحث في الأخبار", value="news")
        language = st.sidebar.selectbox("اختر لغة الأخبار", options=["ar", "en"], index=1)
        page_size = st.sidebar.slider("عدد الأخبار للتحميل", min_value=1, max_value=20, value=5)

        st.markdown("### تصنيف الأخبار من NewsAPI")
        if api_key:
            news_items = self.get_latest_news(api_key, query, language, page_size)
            self.display_news_classification(news_items)
        else:
            st.info("يرجى إدخال مفتاح API لتحميل الأخبار من NewsAPI.")

        st.markdown("---")
        st.markdown("### تصنيف الأخبار من ملف CSV")

        uploaded_file = st.file_uploader("ارفع ملف CSV يحتوي على عمود نص الأخبار", type=["csv"])
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            text_columns = df.columns.tolist()
            text_column = st.selectbox("اختر عمود النص", options=text_columns)
            if st.button("تصنيف الأخبار في الملف"):
                with st.spinner("جاري تصنيف الأخبار..."):
                    result_df = self.classify_csv(df, text_column)
                    st.dataframe(result_df)
                    csv_data = result_df.to_csv(index=False)
                    st.download_button("تحميل الملف بعد التصنيف", data=csv_data, file_name="classified_news.csv", mime="text/csv")

if __name__ == "__main__":
    app = NewsClassifierApp()
    app.run()
