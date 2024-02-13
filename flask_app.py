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
workouts['Exercise_Name'] = workouts['Exercise_Name'].fillna('')
tfidf_matrix = tfidf_vectorizer.fit_transform(workouts['Exercise_Name'])

# Calculate cosine similarity between exercise descriptions
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# Function to recommend exercises based on user preferences
def recommend_exercises(user_preferences, cosine_sim=cosine_sim, workouts=workouts):
    # Filter exercises based on user preferences
    equipment_selected = workouts[workouts['Equipment'] == user_preferences['Equipment']]
    muscleGp_selected = equipment_selected[workouts['muscle_gp'] == user_preferences['Muscle Group']]
    # Exercise_Name,Description_URL,Exercise_Image,Exercise_Image1,muscle_gp_details,muscle_gp,equipment_details,Equipment,Rating,Description

    # Sort exercises by similarity to preferred workout type
    exercise_indices = muscleGp_selected.index
    sim_scores = list(enumerate(cosine_sim[exercise_indices]))
    # sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sorted(sim_scores, reverse=True)

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
        'Muscle Group': request.form['muscle_gp'],
        'Equipment': request.form['equipment']
    }
    recommendations = recommend_exercises(user_preferences)
    return jsonify({'recommendations': recommendations.tolist()})

if __name__ == '__main__':
    app.run(debug=True)


