import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()


def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)


tfidf = pickle.load(open('Model-code/vectorizer.pkl','rb'))
model = pickle.load(open('Model-code/model.pkl','rb'))

# Choose the theme (light or dark)
theme = st.sidebar.selectbox("Choose a theme", ["Light", "Dark"])

# Set the page background color based on the theme
if theme == "Light":
    page_bg_color = "#f7f7f7"
    text_color = "#000000"
else:
    page_bg_color = "#222222"
    text_color = "#ffffff"

# Set the page style
st.markdown(
    f"""
    <style>
    .reportview-container {{
        background-color: {page_bg_color};
        color: {text_color};
    }}
    .reportview-container .main .block-container {{
        background-color: {page_bg_color};
    }}
    .reportview-container .main .block-container .Widget.stTextArea {{
        background-color: {page_bg_color};
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# Set the app title and header
st.title("Email/SMS Spam Classifier")
st.markdown("---")

# Set the input area style
input_sms = st.text_area(
    "Enter the message",
    height=150,
    key="input",
    help="Enter the message",
    )

# Set the button and result area style
btn_bg_color = "#1abc9c"
result_bg_color = "#eafaf1"

if st.button(
    'Predict',
    key="predict",
    help="Click to predict if the message is spam or not",
    ):
    # Preprocess the input text
    transformed_sms = transform_text(input_sms)

    # Vectorize the preprocessed text
    vector_input = tfidf.transform([transformed_sms])

    # Make the prediction
    result = model.predict(vector_input)[0]

    # Display the result
    st.markdown("---")
    if result == 1:
        st.header("🛑 Spam")
    else:
        st.header("✅ Not Spam")
