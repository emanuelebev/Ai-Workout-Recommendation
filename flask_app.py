from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import os

app = Flask(__name__)

# Read the dataset from the "dataset" directory
dataset_path = os.path.join("dataset", "exercise_data.csv")
workouts = pd.read_csv(dataset_path)

# Preprocess the exercise descriptions using TF-IDF vectorization
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
workouts['Description'] = workouts['Description'].fillna('')
tfidf_matrix = tfidf_vectorizer.fit_transform(workouts['Description'])

# Calculate cosine similarity between exercise descriptions
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# Function to recommend exercises based on user preferences
def recommend_exercises(user_preferences, cosine_sim=cosine_sim, workouts=workouts):
    # Filter exercises based on user preferences
    preferred_exercises = workouts[workouts['Workout'] == user_preferences['Preferred Workout Type']]

    # Sort exercises by similarity to preferred workout type
    exercise_indices = preferred_exercises.index
    sim_scores = list(enumerate(cosine_sim[exercise_indices]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get top recommended exercises
    top_exercises_indices = [i[0] for i in sim_scores[:3]]  # Change 3 to the desired number of recommendations
    top_exercises = workouts.iloc[top_exercises_indices]['Exercise_Name']

    return top_exercises

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recommendations', methods=['POST'])
def get_recommendations():
    user_preferences = {
        'Goal': request.form['goal'],
        'Experience Level': request.form['experience'],
        'Preferred Workout Type': request.form['workout_type']
    }
    recommendations = recommend_exercises(user_preferences)
    return jsonify({'recommendations': recommendations.tolist()})

if __name__ == '__main__':
    app.run(debug=True)





# # Read the CSV file into a pandas DataFrame
# df = pd.read_csv("exercises.csv")

# def filter_exercises(df, muscle_group=None, equipment=None, min_rating=None):
#     filtered_df = df.copy()
    
#     if muscle_group:
#         filtered_df = filtered_df[filtered_df['muscle_gp'] == muscle_group]
    
#     if equipment:
#         filtered_df = filtered_df[filtered_df['Equipment'] == equipment]
    
#     if min_rating:
#         filtered_df = filtered_df[filtered_df['Rating'] >= min_rating]
    
#     return filtered_df

# def recommend_exercises(df, muscle_group=None, equipment=None, min_rating=None):
#     filtered_df = filter_exercises(df, muscle_group, equipment, min_rating)
    
#     if len(filtered_df) == 0:
#         return "No exercises found matching the criteria."
    
#     recommended_exercises = filtered_df.sample(min(3, len(filtered_df)))
    
#     return recommended_exercises[['Exercise_Name', 'Description_URL']]

# # Example usage
# recommendations = recommend_exercises(df, muscle_group='Quadriceps', min_rating=9.0)
# print(recommendations)
