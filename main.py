# Import necessary libraries
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Sample workout data (you can replace this with your own dataset)
workouts = pd.DataFrame({
    'Workout': ['Cardio', 'Strength Training', 'Yoga', 'Pilates', 'HIIT'],
    'Description': ['Aerobic exercise for improving cardiovascular health',
                    'Building muscle strength and endurance through resistance exercises',
                    'Mind-body practice for flexibility, strength, and relaxation',
                    'Low-impact exercises focusing on core strength and flexibility',
                    'High-intensity interval training for burning fat and improving fitness']
})

# Sample user data (you can replace this with user input)
user_preferences = {
    'Goal': 'Weight loss',
    'Experience Level': 'Beginner',
    'Preferred Workout Type': 'Cardio'
}

# Preprocess the workout descriptions using TF-IDF vectorization
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
workouts['Description'] = workouts['Description'].fillna('')
tfidf_matrix = tfidf_vectorizer.fit_transform(workouts['Description'])

# Calculate cosine similarity between workout descriptions
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# Function to recommend workouts based on user preferences
def recommend_workouts(user_preferences, cosine_sim=cosine_sim, workouts=workouts):
    # Filter workouts based on user preferences
    preferred_workouts = workouts[workouts['Workout'] == user_preferences['Preferred Workout Type']]
    
    # Sort workouts by similarity to preferred workout type
    workout_indices = preferred_workouts.index
    sim_scores = list(enumerate(cosine_sim[workout_indices]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # Get top recommended workouts
    top_workouts_indices = [i[0] for i in sim_scores[:3]]  # Change 3 to the desired number of recommendations
    top_workouts = workouts.iloc[top_workouts_indices]['Workout']
    
    return top_workouts

# Get personalized workout recommendations for the user
recommendations = recommend_workouts(user_preferences)
print("Personalized Workout Recommendations:")
print(recommendations)
