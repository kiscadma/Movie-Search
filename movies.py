import pandas as pd, numpy as np, regex as re
from tqdm import tqdm
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer


stop_words = set(stopwords.words('english'))
lem = WordNetLemmatizer()
def preprocess(text):
    text = re.split('\W+', text.lower())
    for i in range(len(text)):
        if text[i] in stop_words:
            text[i] = ''
        else:
            text[i] = lem.lemmatize(text[i])
    return ' '.join([t for t in text if t != ''])
    

def load_data(load_processed_file=False):
    '''
    read the csv data file, skipping the first 8796 rows. We start from row 8797 because it is 
    the first Godfather movie, which is iconic. 17378 is the final American movie, so we will 
    stop there to keep the dataframe smaller

    dataset was downloaded from https://www.kaggle.com/jrobischon/wikipedia-movie-plots
    '''
    if load_processed_file:
        data = pd.read_csv("archive/processed_wiki_movie.csv")
    else:
        data = pd.read_csv("archive/wiki_movie_plots_deduped.csv", nrows=17378-8796, 
                        usecols=["Title", "Plot"], skiprows=[i for i in range(1, 8796)])
        data["Plot"] = data.Plot.apply(lambda x: preprocess(x))
        data["Title"] = data.Title.apply(lambda x: x.strip())
        pd.DataFrame.to_csv(data, "archive/processed_wiki_movie.csv", index=False)
    return data


# Load the data, and input plot data to the TFIDF vectorizer.
data = load_data(load_processed_file=True)

# vectorize the data, excluding words that appear in 67% of documents because they don't add much
vectorizer = TfidfVectorizer(max_df=0.67)
tfidf = vectorizer.fit_transform(data["Plot"])
vocab = vectorizer.get_feature_names()
movies = defaultdict(set)

# iterate through movies and find the top words from the TF-IDF, and store them as a set
for i in tqdm(range(data.shape[0]), desc="Finding most significant words from each movie"):
    ind = np.argpartition(tfidf[i,].toarray()[0], kth=-30)[-30:]
    movies[data.at[i, 'Title']] = set([(vocab[j], tfidf[i,j]) for j in ind])

movie_count = 5
while True:
    text = input("\nEnter a movie title or a description of a movie, or 'quit' to stop: ")
    if text == 'quit':
        break
    plot = set(preprocess(text).split()) # preprocess the query to match plots (lemmatized)
    title = set(text.lower().split()) - stop_words # not lemmatized to match exact wording of title
    share_plot = []
    share_title = []

    for m in movies:
        s = set(m.lower().split())
        shared = len(s & title)
        if shared: # add titles that share words
            share_title.append((shared, m))

        score = 0
        for word, val in movies[m]: # words that are less common will have higher vals, let's account for this
            if word in plot:
                score += val
        if score: # add plots that share words
            share_plot.append((score, m))

    share_title.sort(reverse=True)
    print(f"Movie titles most similar to '{text}':")
    for i in range(min(movie_count, len(share_title))):
        print(f"\t{i+1}. {share_title[i][1]}")
        
    share_plot.sort(reverse=True)
    print(f"Movie plots most similar to '{text}':")
    for i in range(min(movie_count, len(share_plot))):
        print(f"\t{i+1}. {share_plot[i][1]}")

