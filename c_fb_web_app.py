import streamlit as st
import pandas as pd
from transformers import pipeline
import spacy
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import numpy as np
import plotly.express as px
import streamlit as st
import spacy
import os

# Ensure the model is installed and then load it
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    # If the model is not found, try downloading it again
    st.warning("Model not found. Attempting to download...")
    os.system('python -m spacy download en_core_web_sm')
    nlp = spacy.load("en_core_web_sm")  # Load it again after download

st.title('🔍 Decoding Customer Sentiments ')
st.write('***')
# Sidebar for file upload
st.sidebar.title("File Operations")
st.sidebar.write('***')
uploaded_file = st.sidebar.file_uploader("Upload a CSV file with customer feedback", type=["csv"])

# Check if a file is uploaded
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)  # Read the uploaded CSV file
    review_column = st.sidebar.selectbox("Select the column that contains reviews:", df.columns)  # Select review column

    st.write("### *Preview of the uploaded data:*")
    st.write(df[[review_column]].head())  # Show the first few rows of the selected column

    # Initialize sentiment analysis pipeline
    sentiment_pipeline = pipeline("sentiment-analysis")

    # Function to analyze sentiment of feedback
    def analyze_sentiment(feedback):
        return sentiment_pipeline(feedback)[0]

    # Perform sentiment analysis
    if review_column:
        try:
            df['sentiment'] = df[review_column].apply(analyze_sentiment)
        except RuntimeError:
            st.warning("Some feedback entries are too long for our system to process. Please ensure each entry is under 512 characters.")

        # Preprocess text for word cloud
        nlp = spacy.load("en_core_web_sm")

        def preprocess_text(text):
            doc = nlp(text)
            return ' '.join([token.lemma_ for token in doc if not token.is_stop and not token.is_punct])

        df['processed_feedback'] = df[review_column].apply(preprocess_text)

        # Create Overall Word Cloud
        st.write("***")
        st.write('### *Overall Word Cloud:*')
        st.write(' ')
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(df['processed_feedback']))
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        st.pyplot(plt)

        # Sentiment Distribution
        sentiment_count = df['sentiment'].apply(lambda x: x['label']).value_counts().reset_index()
        sentiment_count.columns = ['Sentiment', 'Count']
        st.write(' ')
        st.write(' ')
        st.write('### *Sentiment Analysis Charts*')
        fig = px.bar(sentiment_count, x='Sentiment', y='Count', text='Count',title="Sentiment Analysis")
        fig.update_layout(title_x=0.5)
        st.plotly_chart(fig)

        # Create tabs for sentiment analysis
        tab1, tab2, tab3 = st.tabs(["Positive Words", "Negative Words", "Neutral Words"])

        # Function to display word cloud and word counts for each sentiment
        def display_wordcloud_and_barchart(sentiment_label, tab):
            with tab:
                filtered_feedback = df[df['sentiment'].apply(lambda x: x['label'] == sentiment_label)]['processed_feedback']
                if not filtered_feedback.empty:
                    # Word Cloud
                    st.write(f"### *Word cloud for {sentiment_label} feedback:*")
                    st.write('')
                    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(filtered_feedback))
                    plt.figure(figsize=(10, 5))
                    plt.imshow(wordcloud, interpolation='bilinear')
                    plt.axis('off')
                    st.pyplot(plt)

                    # Paired Words Count
                    vectorizer = CountVectorizer(ngram_range=(2, 2))
                    X = vectorizer.fit_transform(filtered_feedback)
                    word_freq = np.asarray(X.sum(axis=0)).flatten()
                    words = vectorizer.get_feature_names_out()
                    top_word_indices = word_freq.argsort()[-10:][::-1]
                    top_words = [words[i] for i in top_word_indices]
                    top_word_counts = [word_freq[i] for i in top_word_indices]
                    if top_word_counts:
                        # Update top words heading
                        st.write('')
                        st.write(f"### *Below word conveys {sentiment_label} connotation:*")
                        fig = px.bar(x=top_words, y=top_word_counts)
                        st.plotly_chart(fig)
                    else:
                        st.warning(f"No paired words found for {sentiment_label}.")
                else:
                    st.warning(f"No feedback found for {sentiment_label}.")

        # Display for each sentiment
        display_wordcloud_and_barchart("POSITIVE", tab1)
        display_wordcloud_and_barchart("NEGATIVE", tab2)
        display_wordcloud_and_barchart("NEUTRAL", tab3)

        # Recommendations based on sentiment
        def generate_recommendations(sentiment):
            if sentiment == 'POSITIVE':
                return "Keep up the good work!"
            elif sentiment == 'NEGATIVE':
                return "Consider improving the areas mentioned in the feedback."
            else:
                return "Look into the feedback for more insights."

        df['recommendations'] = df['sentiment'].apply(lambda x: generate_recommendations(x['label']))

        # Download button
        csv = df[[review_column, 'recommendations']].to_csv(index=False)
        st.sidebar.download_button(label="📥 Download", data=csv, file_name='recommendations.csv', mime='text/csv')

else:
    st.warning("Please upload a CSV file in the left sidebar to proceed.")
