AutoGrader Pro+ 🧠

AutoGrader Pro+ is a powerful AI-powered exam evaluation tool that automatically grades student answers from PDFs, provides semantic scoring, keyword-based evaluation, MCQ explanations, and interactive performance dashboards.

Table of Contents

Features

Installation

Usage

Grading Methods

AI Tutor

Dependencies

License

Features

Upload Answer Key PDFs and parse questions automatically.

Upload multiple Student Answer PDFs for automated grading.

Supports MCQs and descriptive answers.

Semantic similarity scoring using TF-IDF and AI feedback.

Keyword matching grading for concise evaluation.

Optional AI explanations for MCQs.

Interactive results dashboard with:

Total scores

Detailed feedback per question

Downloadable CSV of results

Class performance visualization (Bar, Pie, Box, Histogram, Heatmap)

Built-in AI Doubt Solver to answer student queries instantly.

Sleek dark-themed Streamlit UI.

Installation

Clone the repository:

git clone https://github.com/HARI-45-FAV/AI_EXAM_EVALUTION.git
cd AI_EXAM_EVALUTION


Create and activate a virtual environment:

python -m venv venv
# Windows
.\venv\Scripts\activate
# macOS/Linux
source venv/bin/activate


Install dependencies:

pip install -r requirements.txt


Make sure Ollama is installed and running (required for AI feedback & tutor):

ollama start

Usage

Run the Streamlit app:

streamlit run app.py


Step 1: Upload the Answer Key PDF.

Step 2: Upload Student PDFs and click Run Grading.

Step 3: View results, download CSV, and analyze performance.

Ask the AI Doubt Solver for any exam-related question.

Grading Methods

Semantic Similarity + AI Feedback
Uses TF-IDF cosine similarity and AI-generated feedback to grade descriptive answers.

Keyword Matching
Grades based on matched keywords between student answer and answer key.

MCQs
Automatically grades multiple-choice questions and optionally provides AI explanations.

AI Tutor

Ask any exam or topic-related question.

The AI Tutor responds with concise explanations using the Ollama model.

Dependencies

Python 3.10+

Streamlit

PyMuPDF (fitz)

Ollama Python Client

scikit-learn

pandas

plotly
