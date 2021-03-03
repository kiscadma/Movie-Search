# movie-search
Search for movies using keywords or movie titles

Use the movie_notebook.ipynb jupyer notebook for a guide through the project.


This project uses movies from the Wikipedia movie plot dataset (https://www.kaggle.com/jrobischon/wikipedia-movie-plots), specifically American films from 1972-2017.

Plots are analyzed and use TF-IDF to find significant words, and are stored for lookup. The user can input movie titles or keywords into the console, and the program will print relevant movie titles.
The movies used can be updated in the load_data function, as well as the option to load a processed version of data after the program has been run once.
