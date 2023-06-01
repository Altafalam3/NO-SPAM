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


tfidf = pickle.load(open('Model-code/vectorizer.pkl', 'rb'))
model = pickle.load(open('Model-code/model.pkl', 'rb'))

st.title("Email/SMS Spam Classifier")

input_sms = st.text_area("Enter the message", height=150,
                         key="input",
                         help="Enter the message",
                         )

if st.button('Predict', key="predict", help="Click to predict if the message is spam or not"):
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

# Apply CSS to change the button outline color
st.markdown(
    """
    <style>
    .stButton button {
        border-color: red !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
