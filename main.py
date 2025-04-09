from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import os

app = FastAPI()

# Allow all origins (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Prediction function
def predict_box_office(budget, runtime, popularity, genre):
    genre_bonus = {
        'action': 20,
        'comedy': 10,
        'drama': 5,
        'sci-fi': 25
    }.get(genre.lower(), 0)
    return 2 * budget + 0.5 * runtime + 3 * popularity + genre_bonus

# Pydantic model
class Movie(BaseModel):
    budget: float
    runtime: float
    popularity: float
    genre: str

# HTML page route
@app.get("/", response_class=HTMLResponse)
def get_ui():
    return HTMLResponse(content="""
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Box Office Predictor 🎬</title>
  <style>
    body {
      font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
      background: #f4f6f8;
      padding: 40px 20px;
      max-width: 420px;
      margin: auto;
      color: #333;
    }

    h2 {
      text-align: center;
      color: #222;
      margin-bottom: 20px;
    }

    input, select, button {
      width: 100%;
      padding: 12px;
      margin-top: 12px;
      border: 1px solid #ccc;
      border-radius: 6px;
      box-sizing: border-box;
      font-size: 16px;
    }

    button {
      background-color: #4CAF50;
      color: white;
      font-weight: bold;
      border: none;
      cursor: pointer;
      transition: background-color 0.3s;
    }

    button:hover {
      background-color: #45a049;
    }

    h3#result {
      text-align: center;
      margin-top: 20px;
      font-size: 20px;
      color: #1a73e8;
    }

    select {
      background-color: #fff;
    }
  </style>
</head>
<body>
  <h2>Box Office Predictor 🎬</h2>

  <input type="number" id="budget" placeholder="Budget (in millions)">
  <input type="number" id="runtime" placeholder="Runtime (minutes)">
  <input type="number" id="popularity" placeholder="Popularity Score">

  <select id="genre">
    <option value="action">Action</option>
    <option value="comedy">Comedy</option>
    <option value="drama">Drama</option>
    <option value="sci-fi">Sci-Fi</option>
    <option value="other">Other</option>
  </select>

  <button onclick="predict()">Predict</button>

  <h3 id="result"></h3>

  <script>
    async function predict() {
      const data = {
        budget: parseFloat(document.getElementById('budget').value),
        runtime: parseFloat(document.getElementById('runtime').value),
        popularity: parseFloat(document.getElementById('popularity').value),
        genre: document.getElementById('genre').value
      };

      const response = await fetch("/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(data)
      });

      const result = await response.json();
      document.getElementById('result').innerText = 
        "🎯 Predicted Box Office: $" + result.box_office_prediction + "M";
    }
  </script>
</body>
</html>
""", media_type="text/html")

# Prediction endpoint
@app.post("/predict")
def predict(movie: Movie):
    prediction = predict_box_office(
        movie.budget, movie.runtime, movie.popularity, movie.genre
    )
    return {"box_office_prediction": round(prediction, 2)}

# Entry point
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
