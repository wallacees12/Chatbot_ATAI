import pandas as pd

# Define file paths
title_basics_path = 'title.basics.tsv'   # Path to title basics file
title_ratings_path = 'title.ratings.tsv.gz' # Path to title ratings file

# Load the title basics data with only necessary columns
movies_df = pd.read_csv(title_basics_path, sep='\t', usecols=['tconst', 'titleType', 'primaryTitle'], low_memory=False)

# Filter to keep only rows where titleType is 'movie'
movies_df = movies_df[movies_df['titleType'] == 'movie']

# Load the title ratings data
ratings_df = pd.read_csv(title_ratings_path, sep='\t', low_memory=False)

# Merge movies and ratings data on 'tconst'
merged_df = pd.merge(movies_df, ratings_df, on='tconst')

# Sort by number of votes (numVotes) in descending order to get the most popular movies
top_movies_by_votes = merged_df.sort_values(by='numVotes', ascending=False).head(10000)

# Extract the movie titles
movie_titles = top_movies_by_votes['primaryTitle'].tolist()

# Save the top 10,000 movie titles to a text file
with open('top_10000_movies_by_votes.txt', 'w') as f:
    for title in movie_titles:
        f.write(f"{title}\n")

print("Top 10,000 movie titles by popularity (number of votes) have been saved to 'top_10000_movies_by_votes.txt'.")