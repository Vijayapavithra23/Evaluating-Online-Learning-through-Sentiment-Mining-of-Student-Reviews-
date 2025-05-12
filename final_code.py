# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

# Download NLTK resources
nltk.download(['punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger', 'vader_lexicon'])

# Load dataset (replace with your Kaggle dataset)
# If reading CSV, check headers
df = pd.read_csv('online_class_feedback.csv', header=0)  # Ensure correct header row # Columns: 'feedback', 'sentiment' (if labeled)
print(df.head())
# Initialize tools
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    # Lowercase
    text = text.lower()
    # Tokenize
    tokens = word_tokenize(text)
    # Remove stopwords & non-alphabetic
    tokens = [word for word in tokens if word.isalpha() and word not in stop_words]
    # POS Tagging + Lemmatization
    pos_tags = pos_tag(tokens)
    tokens = [lemmatizer.lemmatize(word, pos=get_wordnet_pos(tag)) for word, tag in pos_tags]
    return ' '.join(tokens)

def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'): return 'a'  # Adjective
    elif treebank_tag.startswith('V'): return 'v'  # Verb
    elif treebank_tag.startswith('N'): return 'n'  # Noun
    elif treebank_tag.startswith('R'): return 'r'  # Adverb
    else: return 'n'  # Default: Noun

# Apply preprocessing
df['cleaned_feedback'] = df['feedback'].apply(preprocess_text)
print("\nSample Preprocessed Text:")
print(df['cleaned_feedback'].head())
# Initialize VADER sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

def get_sentiment(text):
    scores = analyzer.polarity_scores(text)
    if scores['compound'] >= 0.05: return 'positive'
    elif scores['compound'] <= -0.05: return 'negative'
    else: return 'neutral'

df['sentiment'] = df['cleaned_feedback'].apply(get_sentiment)

# Visualize sentiment distribution
plt.figure(figsize=(8, 5))
sns.countplot(x='sentiment', data=df, palette='viridis')
plt.title("Sentiment Distribution in Student Feedback")
plt.savefig("sentiment_distribution.png")
plt.show()
# If dataset has labels ('positive', 'negative', 'neutral')
if 'sentiment' in df.columns:
    # TF-IDF Vectorization
    tfidf = TfidfVectorizer(max_features=5000)
    X = tfidf.fit_transform(df['cleaned_feedback'])
    y = df['sentiment']
    
    # Split data (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train models
    models = {
        "Naive Bayes": MultinomialNB(),
        "SVM": SVC(kernel='linear'),
        "Random Forest": RandomForestClassifier()
    }
    
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        results[name] = accuracy
        
        # Print classification report
        print(f"\n{name} Performance:")
        print(classification_report(y_test, y_pred))
        
        # Confusion Matrix
        plt.figure(figsize=(6, 4))
        sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
        plt.title(f"{name} Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.savefig(f"{name.lower().replace(' ', '_')}_confusion_matrix.png")
        plt.show()
    
    # Compare model accuracy
    plt.figure(figsize=(8, 5))
    plt.bar(results.keys(), results.values(), color=['blue', 'green', 'red'])
    plt.title("Model Accuracy Comparison")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.savefig("model_accuracy_comparison.png")
    plt.show()
    # Word Cloud for Positive Feedback
positive_text = ' '.join(df[df['sentiment'] == 'positive']['cleaned_feedback'])
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(positive_text)
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title("Word Cloud: Positive Feedback")
plt.savefig("positive_wordcloud.png")
plt.show()

# Word Cloud for Negative Feedback
negative_text = ' '.join(df[df['sentiment'] == 'negative']['cleaned_feedback'])
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(negative_text)
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title("Word Cloud: Negative Feedback")
plt.savefig("negative_wordcloud.png")
plt.show()

# Export to Excel for further analysis
df.to_excel("processed_feedback_results.xlsx", index=False)
print("\n✅ Analysis complete! Results saved in Excel and visualizations exported.")