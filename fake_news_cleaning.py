import streamlit as st
import requests
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from translate import Translator
from transformers import pipeline

nltk.download('stopwords')

translator = Translator(to_lang="en", from_lang="ar")
classifier = pipeline("text-classification", model="mrm8488/bert-tiny-finetuned-fake-news")

stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    words = text.split()
    words = [word for word in words if word not in stop_words]
    words = [stemmer.stem(word) for word in words]
    return ' '.join(words)

def get_latest_news(api_key, query="news", language="en", page_size=5):
    url = (
        'https://newsapi.org/v2/everything?'
        f'q={query}&'
        f'language={language}&'
        f'pageSize={page_size}&'
        'sortBy=publishedAt&'
        f'apiKey={api_key}'
    )
    response = requests.get(url)
    data = response.json()
    if data['status'] == 'ok':
        articles = data['articles']
        return [article['title'] + ". " + article.get('description', '') for article in articles]
    else:
        st.error("Error fetching news: " + data.get('message', 'Unknown error'))
        return []

def classify_arabic_news(text):
    try:
        translated = translator.translate(text)
        result = classifier(translated)[0]
        label = result['label']
        score = result['score']
        final_label = "خبر حقيقي ✅" if label.lower() == "real" else "خبر كاذب ❌"
        return final_label, translated, score
    except Exception as e:
        return f"Error: {e}", "", 0

# Streamlit interface
st.set_page_config(page_title="NewsTruth AI", layout="wide")
st.title("📰 NewsTruth AI – تصنيف الأخبار العربية والإنجليزية")

api_key = st.text_input("🔑 أدخل مفتاح NewsAPI الخاص بك:", value="", type="password")

if api_key:
    query = st.text_input("🔍 اكتب كلمة بحث (مثال: سوريا، سياسة، لقاح):", value="Syria OR vaccine")

    if st.button("📡 جلب وتصنيف الأخبار الحديثة"):
        lang = "ar" if any('\u0600' <= c <= '\u06FF' for c in query) else "en"
        news_items = get_latest_news(api_key, query="Syria OR vaccine", language="en", page_size=5)

        st.subheader("🗞️ الأخبار المصنفة:")
        for i, news in enumerate(news_items, 1):
            if lang == "ar":
                label, translated, score = classify_arabic_news(news)
            else:
                result = classifier(news)[0]
                label = "خبر حقيقي ✅" if result['label'].lower() == "real" else "خبر كاذب ❌"
                translated, score = "", result['score']

            st.markdown(f"**{i}. الخبر:** {news}")
            if translated:
                st.markdown(f"*الترجمة:* {translated}")
            st.markdown(f"🔎 **النتيجة:** {label} (الثقة: {score:.2f})")
            st.write("---")

    st.subheader("✍️ تصنيف خبر عربي يدوي:")
    user_input = st.text_area("أدخل نص الخبر هنا:")

    if st.button("تحليل الخبر"):
        if user_input.strip():
            label, translated, score = classify_arabic_news(user_input)
            st.markdown(f"**🔄 الترجمة:** {translated}")
            st.markdown(f"**🔍 التصنيف:** {label} (الثقة: {score:.2f})")
        else:
            st.warning("يرجى إدخال نص أولاً.")

# Sidebar
st.sidebar.markdown("## 👤 Riad Karkoura")
st.sidebar.markdown("صحفي تقني | مختص بالذكاء الاصطناعي والتحقق من الأخبار")
st.sidebar.markdown("[🔗 LinkedIn](https://www.linkedin.com/in/riad-karkoura-b9010b196)")
