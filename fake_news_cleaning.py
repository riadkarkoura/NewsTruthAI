import streamlit as st
import requests
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from deep_translator import GoogleTranslator
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import csv

nltk.download('stopwords')

translator = GoogleTranslator(source='auto', target='en')
analyzer = SentimentIntensityAnalyzer()

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

def classify_text_with_vader(text):
    try:
        translated = translator.translate(text)
        scores = analyzer.polarity_scores(translated)
        compound = scores['compound']
        # اذا النتيجة موجبة فهي ايجابية (خبر حقيقي)، اذا سالبة فهي خبر كاذب
        if compound >= 0.05:
            label = "خبر حقيقي ✅"
        elif compound <= -0.05:
            label = "خبر كاذب ❌"
        else:
            label = "خبر غير واضح 🤔"
        return label, translated, compound
    except Exception as e:
        return f"Error: {str(e)}", "", 0

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
            label, translated, score = classify_text_with_vader(news)

            st.markdown(f"**{i}. الخبر:** {news}")
            if translated:
                st.markdown(f"*الترجمة:* {translated}")
            st.markdown(f"🔎 **النتيجة:** {label} (درجة الثقة: {score:.2f})")

            col1, col2 = st.columns(2)
            with col1:
                if st.button(f"✅ التصنيف صحيح (خبر {i})"):
                    save_feedback(news, label, "correct")
                    st.success("تم حفظ التقييم كتصنيف صحيح.")
            with col2:
                if st.button(f"❌ التصنيف خاطئ (خبر {i})"):
                    save_feedback(news, label, "wrong")
                    st.warning("تم حفظ التقييم كتصنيف خاطئ.")
            st.write("---")

    st.subheader("✍️ تصنيف خبر عربي يدوي:")
    user_input = st.text_area("أدخل نص الخبر هنا:")

    if st.button("تحليل الخبر"):
        if user_input.strip():
            label, translated, score = classify_text_with_vader(user_input)
            st.markdown(f"**🔄 الترجمة:** {translated}")
            st.markdown(f"**🔍 التصنيف:** {label} (درجة الثقة: {score:.2f})")

            col1, col2 = st.columns(2)
            with col1:
                if st.button("✅ التصنيف صحيح (الخبر اليدوي)"):
                    save_feedback(user_input, label, "correct")
                    st.success("تم حفظ التقييم كتصنيف صحيح.")
            with col2:
                if st.button("❌ التصنيف خاطئ (الخبر اليدوي)"):
                    save_feedback(user_input, label, "wrong")
                    st.warning("تم حفظ التقييم كتصنيف خاطئ.")
        else:
            st.warning("يرجى إدخال نص أولاً.")

st.sidebar.markdown("## 👤 Riad Karkoura")
st.sidebar.markdown("صحفي تقني | مختص بالذكاء الاصطناعي والتحقق من الأخبار")
st.sidebar.markdown("[🔗 LinkedIn](https://www.linkedin.com/in/riad-karkoura-b9010b196)")
