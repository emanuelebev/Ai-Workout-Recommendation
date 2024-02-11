import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
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

# Function to handle button click event
def get_recommendations():
    user_preferences = {
        'Goal': goal_combobox.get(),
        'Experience Level': experience_combobox.get(),
        'Preferred Workout Type': workout_combobox.get()
    }
    recommendations = recommend_workouts(user_preferences)
    messagebox.showinfo("Recommendations", f"Personalized Workout Recommendations: {', '.join(recommendations)}")

# Create main window
root = tk.Tk()
root.title("Workout Recommendation System")

# Create and add widgets to the main window
goal_label = tk.Label(root, text="Fitness Goal:")
goal_label.grid(row=0, column=0, padx=10, pady=5, sticky="w")
goals = ['Weight Loss', 'Muscle Building', 'Flexibility', 'Stress Relief']
goal_combobox = ttk.Combobox(root, values=goals, state="readonly")
goal_combobox.current(0)
goal_combobox.grid(row=0, column=1, padx=10, pady=5)

experience_label = tk.Label(root, text="Experience Level:")
experience_label.grid(row=1, column=0, padx=10, pady=5, sticky="w")
experiences = ['Beginner', 'Intermediate', 'Advanced']
experience_combobox = ttk.Combobox(root, values=experiences, state="readonly")
experience_combobox.current(0)
experience_combobox.grid(row=1, column=1, padx=10, pady=5)

workout_label = tk.Label(root, text="Preferred Workout Type:")
workout_label.grid(row=2, column=0, padx=10, pady=5, sticky="w")
workout_types = ['Cardio', 'Strength Training', 'Yoga', 'Pilates', 'HIIT']
workout_combobox = ttk.Combobox(root, values=workout_types, state="readonly")
workout_combobox.current(0)
workout_combobox.grid(row=2, column=1, padx=10, pady=5)

recommend_button = tk.Button(root, text="Get Recommendations", command=get_recommendations)
recommend_button.grid(row=3, column=0, columnspan=2, padx=10, pady=10)

# Run the main event loop
root.mainloop()
