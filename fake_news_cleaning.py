import streamlit as st
import requests
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score
from translate import Translator  # مكتبة الترجمة
import nltk
nltk.download('stopwords')

# إعداد الترجمة من العربية إلى الإنجليزية
translator = Translator(to_lang="en", from_lang="ar")

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

# قراءة البيانات
df = pd.read_csv("liar_sample.csv")
label_encoder = LabelEncoder()
df['label_encoded'] = label_encoder.fit_transform(df['label'])
df['clean_text'] = df['text'].apply(clean_text)

X_train, X_test, y_train, y_test = train_test_split(
    df['clean_text'], df['label_encoded'], test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# دالة لجلب الأخبار الحديثة من NewsAPI
def get_latest_news(api_key, query="news", language="en", page_size=5):
    url = ('https://newsapi.org/v2/everything?'
           f'q={query}&'
           f'language={language}&'
           f'pageSize={page_size}&'
           'sortBy=publishedAt&'
           f'apiKey={api_key}')
    response = requests.get(url)
    data = response.json()
    if data['status'] == 'ok':
        articles = data['articles']
        news_texts = [article['title'] + ". " + article.get('description', '') for article in articles]
        return news_texts
    else:
        st.error("Error fetching news: " + data.get('message', 'Unknown error'))
        return []

def classify_arabic_news(arabic_text):
    try:
        translated_text = translator.translate(arabic_text)
        cleaned = clean_text(translated_text)
        vect = vectorizer.transform([cleaned])
        pred_encoded = model.predict(vect)[0]
        pred_label = label_encoder.inverse_transform([pred_encoded])[0]
        return pred_label, translated_text
    except Exception as e:
        return f"Error: {e}", ""

# ---- Streamlit UI ----

st.title("تصنيف الأخبار مع دعم اللغة العربية والإنجليزية")

api_key = st.text_input("أدخل مفتاح NewsAPI الخاص بك:", value="cb432eea97dd4cec984e6917dae798bf", type="password")

if api_key:
    query = st.text_input("اكتب كلمة البحث لجلب الأخبار الحديثة (بالإنجليزية أو العربية):", value="Syria OR vaccine OR politics")

    if st.button("جلب الأخبار الحديثة وتصنيفها"):
        lang = "en"
        if any('\u0600' <= c <= '\u06FF' for c in query):  # تحقق لو النص يحتوي عربي
            lang = "ar"

        # إذا كان البحث عربي، نحتاج نترجم الأخبار لاحقًا بعد جلبها
        if lang == "en":
            latest_news = get_latest_news(api_key, query=query, language=lang, page_size=5)
        else:
            # جلب الأخبار بالإنجليزية فقط (NewsAPI لا يدعم العربية بشكل رسمي) ونفترض البحث الإنجليزي مكافئ هنا
            latest_news = get_latest_news(api_key, query="Syria OR vaccine OR politics", language="en", page_size=5)

        st.subheader("الأخبار الحديثة وتصنيفها:")
        for i, news in enumerate(latest_news, 1):
            if lang == "ar":
                # إذا أردنا تصنيف بالعربية: نترجم الخبر أولا ثم نصنف
                label, translated = classify_arabic_news(news)
            else:
                cleaned_news = clean_text(news)
                vect_news = vectorizer.transform([cleaned_news])
                pred_encoded_news = model.predict(vect_news)[0]
                label = label_encoder.inverse_transform([pred_encoded_news])[0]
                translated = ""

            st.markdown(f"**{i}. الخبر:** {news}")
            if translated:
                st.markdown(f"**ترجمة (إنجليزية):** {translated}")
            st.markdown(f"**تصنيف:** {label}")
            st.write("---")

    st.subheader("تصنيف خبر عربي مباشر:")
    user_arabic_input = st.text_area("أدخل نص خبر عربي هنا:")

    if st.button("تصنيف الخبر العربي"):
        if user_arabic_input.strip() == "":
            st.warning("يرجى إدخال نص الخبر العربي أولاً.")
        else:
            label, translated = classify_arabic_news(user_arabic_input)
            st.markdown(f"**النص المترجم:** {translated}")
            st.markdown(f"**تصنيف الخبر:** {label}")

