from flask import Flask, render_template, request, session, redirect, url_for
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import os

app = Flask(__name__)
app.secret_key = "lavender_smart_shop"

# Load Data
# This ensures the app finds the CSV in the same folder as app.py
csv_path = os.path.join(os.path.dirname(__file__), 'products.csv')
df = pd.read_csv(csv_path)

# AI Vectorization: The "Brain" of your project
tfidf = TfidfVectorizer(stop_words='english')
# We fill empty descriptions with an empty string to prevent crashes
tfidf_matrix = tfidf.fit_transform(df['Description'].fillna(''))
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

def get_recommendations(item_name):
    try:
        # Step 1: Find the item index in our matrix
        idx = df.index[df['Name'] == item_name][0]
        # Step 2: Compute similarity scores
        sim_scores = list(enumerate(cosine_sim[idx]))
        # Step 3: Sort items based on similarity
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        # Step 4: Get top 3 (excluding the item itself)
        indices = [i[0] for i in sim_scores[1:4]]
        return df.iloc[indices].to_dict('records')
    except Exception as e:
        print(f"Error in recommendation logic: {e}")
        return []

@app.route('/')
def home():
    if 'user' in session:
        # Keep the page looking full with 4 random trending items
        trending = df.sample(min(4, len(df))).to_dict('records')
        return render_template('index.html', user=session['user'], trending=trending)
    return render_template('login.html')

@app.route('/login', methods=['POST'])
def login():
    username = request.form.get('username')
    if username:
        session['user'] = username
    return redirect(url_for('home'))

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('home'))

@app.route('/search', methods=['POST'])
def search():
    query = request.form.get('query', '').lower()
    # Search logic across names and categories
    search_results = df[df['Name'].str.lower().str.contains(query, na=False) | 
                        df['Category'].str.lower().str.contains(query, na=False)].to_dict('records')
    
    recs = []
    category_name = "Results"
    if search_results:
        # Trigger the recommendation engine for the first search result
        recs = get_recommendations(search_results[0]['Name'])
        category_name = search_results[0]['Category']
    
    return render_template('index.html', 
                           user=session.get('user'), 
                           search_results=search_results, 
                           recommendations=recs,
                           active_category=category_name)

if __name__ == '__main__':
    app.run(debug=True)