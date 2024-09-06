import pandas as pd # type: ignore
import numpy as np # type: ignore
from flask import Flask, request # type: ignore
import pickle
from pyngrok import ngrok # type: ignore

app = Flask(__name__)

# Sample CSS embedded in the HTML template
css = """
<style>
    body {
         font-family: Arial, sans-serif;
                margin: 0;
                padding: 0;
                background-image: url('https://upload.wikimedia.org/wikipedia/commons/thumb/8/8e/Truckee_Meadows_Community_College_%2810075995964%29.jpg/1200px-Truckee_Meadows_Community_College_%2810075995964%29.jpg'); /* Replace with your background image URL */
                background-size: cover;
                background-position: center;
                display: flex;
                justify-content: center;
                align-items: center;
                height: 100vh;
                color: #333;
    }
    .container {
        max-width: 600px;
        margin: 50px auto;
        padding: 20px;
        background-color: #99ccff;
        box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.1);
    }
    h1 {
        text-align: center;
        color: #343a40;
    }
    label {
        display: block;
        margin-bottom: 10px;
        color: #495057;
    }
    input[type="text"], input[type="number"], select {
        width: 100%;
        padding: 10px;
        margin-bottom: 20px;
        border: 1px solid #ced4da;
        border-radius: 5px;
    }
    input[type="submit"] {
        background-color: #007bff;
        color: white;
        padding: 10px 15px;
        border: none;
        border-radius: 5px;
        cursor: pointer;
        width: 100%;
    }
    input[type="submit"]:hover {
        background-color: #0056b3;
    }
</style>
"""
model_path = 'student_performance_model.pkl'
with open(model_path, 'rb') as file:
    model = pickle.load(file)

# App route to index page
@app.route('/')
def index():
    return f"""
    {css}
    <div class="container">
        <h1>Student Performance Predictor</h1>
        <form action="/predict" method="POST">
            <label for="hours_studied">Hours Studied (max 24 hours):</label>
            <input type="number" name="hours_studied" max="24" required>

            <label for="previous_scores">Previous Scores (max 100):</label>
            <input type="number" name="previous_scores" max="100" required>

            <label for="extracurricular_activities">Extracurricular Activities:</label>
            <select name="extracurricular_activities" required>
                <option value="Yes">Yes</option>
                <option value="No">No</option>
            </select>

            <label for="sleep_hours">Sleep Hours:</label>
            <input type="number" name="sleep_hours" required>

            <label for="sample_question_papers">Sample Question Papers Practiced:</label>
            <input type="number" name="sample_question_papers" required>

            <input type="submit" value="Predict">
        </form>
    </div>
    """

@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    hours_studied = float(request.form['hours_studied'])
    previous_scores = float(request.form['previous_scores'])
    extracurricular_activities = request.form['extracurricular_activities']
    sleep_hours = float(request.form['sleep_hours'])
    sample_question_papers = float(request.form['sample_question_papers'])
    
    #Error messages if inputs are incorrect
    if not (0 <= hours_studied <= 24):
        return "Error: Hours Studied must be between 0 and 24."
    if not (0 <= previous_scores <= 100):
        return "Error: Previous Scores must be between 0 and 100."
    if not (0 <= sleep_hours <= 24 - hours_studied):
        return f"Error: Sleep Hours must be between 0 and {24 - hours_studied}."
    if not (0 <= sample_question_papers):
        return "Error: Sample Question Papers Practiced must be non-negative."

    # Convert 'Extracurricular Activities' to 1/0
    extracurricular_activities = 1 if extracurricular_activities == 'Yes' else 0

    # Create the feature array
    features = np.array([[hours_studied, previous_scores, extracurricular_activities, sleep_hours, sample_question_papers]])

    # Predict the performance index
    prediction = model.predict(features)[0]

    return f"""
    {css}
    <div class="container">
        <h1>Prediction Result</h1>
        <p>Predicted Performance Index: <strong>{prediction:.2f}</strong></p>
        <a href="/">Go back</a>
    </div>
    """

if __name__ == '__main__':

    # Run the Flask app
    app.run(port=5000)