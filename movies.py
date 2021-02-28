import pandas as pd, numpy as np, regex as re
from tqdm import tqdm
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer


stop_words = stopwords.words('english')
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

# vectorize the data, limited to 20k words to prevent movie descriptions from just being names
vectorizer = TfidfVectorizer(max_features=20000)
tfidf = vectorizer.fit_transform(data["Plot"])
vocab = vectorizer.get_feature_names()
movies = defaultdict(list)

# iterate through movies and find the top words from the TF-IDF, and store them as a list
for i in tqdm(range(data.shape[0]), desc="Finding most significant words from each movie"):
    ind = np.argpartition(tfidf[i,].toarray()[0], kth=-25)[-25:]
    movies[data.at[i, 'Title']] = [vocab[j] for j in ind]

while True:
    text = input("\nEnter a movie title or a description of a movie, or 'quit' to stop: ")
    if text == 'quit':
        break
    plot = set(preprocess(text).split()) # preprocess the query to match plots
    title = set(text.lower().split()) # unprocessed text to match title exactly
    share_plot = []
    share_title = []
    for m in movies:
        s = set(m.lower().split())
        if len(s & title): # add titles that share words
            share_title.append((len(s & title), m))
        share_plot.append((len(set(movies[m]) - plot), m))

    share_title.sort(reverse=True)
    print(f"Movie titles most similar to '{text}':")
    for i in range(min(5, len(share_title))):
        print(f"\t{i+1}. {share_title[i][1]}")
        
    share_plot.sort()
    print(f"Movie plots most similar to '{text}':")
    for i in range(min(5, len(share_plot))):
        print(f"\t{i+1}. {share_plot[i][1]}")
