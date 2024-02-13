from flask import Flask, jsonify, render_template, request, send_file, make_response
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
    muscleGp_selected = equipment_selected[equipment_selected['muscle_gp'] == user_preferences['Muscle Group']]

    # Sort exercises by similarity to preferred workout type
    exercise_indices = muscleGp_selected.index
    sim_scores = list(enumerate(cosine_sim[exercise_indices]))
    sim_scores = sorted(sim_scores, reverse=True)

    # Get top recommended exercises with all columns
    top_exercises_indices = [i[0] for i in sim_scores[:5]]  # Change 5 to the desired number of recommendations
    top_exercises = workouts.iloc[top_exercises_indices]

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
    return jsonify({'recommendations': recommendations.values.tolist()})


if __name__ == '__main__':
    app.run(debug=True)
