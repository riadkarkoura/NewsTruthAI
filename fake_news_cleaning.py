import streamlit as st
import requests
import pandas as pd
import re
import nltk
import csv
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from deep_translator import GoogleTranslator
from transformers import pipeline

nltk.download('stopwords')

translator = GoogleTranslator(source='auto', target='en')
classifier = pipeline("text-classification", model="microsoft/xtremedistil-l6-h384-uncased")

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
        final_label = "خبر حقيقي ✅" if label.upper() == "REAL" or label.upper() == "LABEL_1" else "خبر كاذب ❌"
        return final_label, translated, score, label
    except Exception as e:
        return f"Error: {e}", "", 0, ""

def save_feedback(news, predicted_label, feedback):
    with open("feedback.csv", "a", newline='', encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([news, predicted_label, feedback])

st.set_page_config(page_title="NewsTruth AI", layout="wide")
st.title("📰 NewsTruth AI – تصنيف الأخبار العربية والإنجليزية")

api_key = st.text_input("🔑 أدخل مفتاح NewsAPI الخاص بك:", value="", type="password")

if api_key:
    query = st.text_input("🔍 اكتب كلمة بحث (مثال: سوريا، سياسة، لقاح):", value="Syria OR vaccine")

    if st.button("📡 جلب وتصنيف الأخبار الحديثة"):
        lang = "ar" if any('\u0600' <= c <= '\u06FF' for c in query) else "en"
        news_items = get_latest_news(api_key, query=query, language=lang, page_size=5)

        st.subheader("🗞️ الأخبار المصنفة:")
        for i, news in enumerate(news_items, 1):
            if lang == "ar":
                label, translated, score, raw_label = classify_arabic_news(news)
            else:
                result = classifier(news)[0]
                raw_label = result['label']
                label = "خبر حقيقي ✅" if raw_label.upper() == "REAL" or raw_label.upper() == "LABEL_1" else "خبر كاذب ❌"
                translated, score = "", result['score']

            st.markdown(f"**{i}. الخبر:** {news}")
            if translated:
                st.markdown(f"*الترجمة:* {translated}")
            st.markdown(f"🔎 **النتيجة:** {label} (الثقة: {score:.2%})")

            col1, col2 = st.columns(2)
            with col1:
                if st.button(f"✅ التصنيف صحيح (خبر {i})"):
                    save_feedback(news, raw_label, "correct")
                    st.success("تم حفظ التقييم كتصنيف صحيح.")
            with col2:
                if st.button(f"❌ التصنيف خاطئ (خبر {i})"):
                    save_feedback(news, raw_label, "wrong")
                    st.warning("تم حفظ التقييم كتصنيف خاطئ.")
            st.write("---")

    st.subheader("✍️ تصنيف خبر عربي يدوي:")
    user_input = st.text_area("أدخل نص الخبر هنا:")

    if st.button("تحليل الخبر"):
        if user_input.strip():
            label, translated, score, raw_label = classify_arabic_news(user_input)
            st.markdown(f"**🔄 الترجمة:** {translated}")
            st.markdown(f"**🔍 التصنيف:** {label} (الثقة: {score:.2%})")

            col1, col2 = st.columns(2)
            with col1:
                if st.button("✅ التصنيف صحيح (الخبر اليدوي)"):
                    save_feedback(user_input, raw_label, "correct")
                    st.success("تم حفظ التقييم كتصنيف صحيح.")
            with col2:
                if st.button("❌ التصنيف خاطئ (الخبر اليدوي)"):
                    save_feedback(user_input, raw_label, "wrong")
                    st.warning("تم حفظ التقييم كتصنيف خاطئ.")
        else:
            st.warning("يرجى إدخال نص أولاً.")

st.sidebar.markdown("## 👤 Riad Karkoura")
st.sidebar.markdown("صحفي تقني | مختص بالذكاء الاصطناعي والتحقق من الأخبار")
st.sidebar.markdown("[🔗 LinkedIn](https://www.linkedin.com/in/riad-karkoura-b9010b196)")
