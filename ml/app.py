import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Sample dataset (inside code)
texts = [
    "Government launches new scheme for farmers",
    "Aliens landed in India yesterday",
    "Stock market reaches all time high",
    "Celebrity says petrol cures diseases",
    "New education policy announced",
    "Man becomes invisible after magic fruit"
]

labels = ["REAL", "FAKE", "REAL", "FAKE", "REAL", "FAKE"]

# Convert text to numbers
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)

# Train model
model = LogisticRegression()
model.fit(X, labels)

# Streamlit UI
st.title("📰 Fake News Detection (Simple Website)")

st.write("Enter news text to check if it is REAL or FAKE")

# Input
news = st.text_area("Enter News:")

# Button
if st.button("Check News"):
    if news.strip() == "":
        st.warning("Please enter some text")
    else:
        data = vectorizer.transform([news])
        result = model.predict(data)[0]

        if result == "FAKE":
            st.error("❌ Fake News")
        else:
            st.success("✅ Real News")