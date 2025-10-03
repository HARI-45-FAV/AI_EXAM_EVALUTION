# AutoGrader Pro+ 🧠
AI-powered exam grader that automatically evaluates **MCQs and descriptive answers** from PDFs, provides **semantic scoring, keyword grading, AI feedback**, and interactive dashboards.

---

## Features

* Upload **Answer Key PDFs** and **Student PDFs**.
* Supports **MCQs & Descriptive Questions**.
* **Semantic similarity** & **keyword-based scoring**.
* **MCQ explanations** via AI.
* Interactive **dashboard** & **performance charts**.
* **AI Doubt Solver** for instant answers.

---

## Installation

```bash
git clone https://github.com/HARI-45-FAV/AI_EXAM_EVALUTION.git
cd AI_EXAM_EVALUTION
python -m venv venv
# Windows
.\venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
pip install -r requirements.txt
ollama start
streamlit run app.py
```

---

## Usage

1. Upload **Answer Key PDF**.
2. Upload **Student PDFs** and click **Run Grading**.
3. View results, download CSV, and analyze charts.
4. Ask questions with **AI Doubt Solver**.

---

## Grading Methods

* **Semantic Similarity + AI Feedback**
* **Keyword Matching**
* **MCQs with optional AI explanations**
