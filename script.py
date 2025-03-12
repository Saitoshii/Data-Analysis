import re
import nltk
import pandas as pd
from collections import Counter
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

nltk.download('stopwords')
nltk.download('punkt')

### Laden der Tweets aus der CSV-Datei ###
csv_datei = "train.csv"
df = pd.read_csv(csv_datei, encoding="ISO-8859-1") 

# Prüfen, ob die benötigten Spalten vorhanden sind
print("Verfügbare Spalten:", df.columns)

# Tweets & Sentiment extrahieren
tweets = df["SentimentText"].astype(str).tolist()
sentiments = df["Sentiment"].tolist()

print(f"{len(tweets)} Tweets wurden aus der Datei geladen.")

### Datenvorverarbeitung ###
def clean_text(text):
    text = text.lower()  
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)  
    text = re.sub(r"\@\w+", "", text)  
    text = re.sub(r"[^\w\s]", "", text)  

    tokens = word_tokenize(text)  
    stop_words = set(stopwords.words('english'))  
    tokens = [word for word in tokens if word not in stop_words and len(word) > 1]  

    return " ".join(tokens)

# Bereinigung der Tweets
cleaned_tweets = [clean_text(tweet) for tweet in tweets if isinstance(tweet, str)]

# Speichern der bereinigten Daten
df_cleaned = pd.DataFrame({'Sentiment': sentiments, 'Tweet': cleaned_tweets})
df_cleaned.to_csv("tweets_cleaned.csv", index=False)
print("Bereinigte Tweets wurden in 'tweets_cleaned.csv' gespeichert.")

### Entitätsanalyse (Hashtags & Erwähnungen) ###
hashtags = [re.findall(r"#(\w+)", tweet) for tweet in cleaned_tweets]
hashtags_flat = [hashtag for sublist in hashtags for hashtag in sublist]
top_hashtags = Counter(hashtags_flat).most_common(10)

print("Top Hashtags:", top_hashtags)

### Themenanalyse mit LDA ###
if len(cleaned_tweets) > 0:
    vectorizer = CountVectorizer(max_features=1000, stop_words='english')
    tf_matrix = vectorizer.fit_transform(cleaned_tweets)

    lda = LatentDirichletAllocation(n_components=5, random_state=42)
    lda.fit(tf_matrix)

    terms = vectorizer.get_feature_names_out()
    for i, topic in enumerate(lda.components_):
        print(f"\nThema {i+1}:")
        print(" ".join([terms[i] for i in topic.argsort()[:-10 - 1:-1]]))
else:
    print("Keine gültigen Tweets nach der Vorverarbeitung gefunden!")

# Wordcloud zur Visualisierung
if cleaned_tweets:
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(" ".join(cleaned_tweets))
    wordcloud.to_image().show()
