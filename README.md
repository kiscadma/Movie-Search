# movie-search
Search for movies using keywords or movie titles


This project uses movies from the Wikipedia movie plot dataset (https://www.kaggle.com/jrobischon/wikipedia-movie-plots), specifically American ones since 1972.

Plots are analyzed and use TF-IDF to find significant words, and are stored for lookup. The user can input movie titles or keywords into the console, and the program will print relevant movie titles.

Ideally, the project would be enhanced to weight words from the TFIDF, so entries that feature more rare words would appear higher in the output.

