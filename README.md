
# AutoGrader Pro+ 🧠

**AutoGrader Pro+** is a powerful AI-powered exam evaluation tool that automatically grades student answers from PDFs, provides semantic scoring, keyword-based evaluation, MCQ explanations, and interactive performance dashboards.

---

## Table of Contents

* [Features](#features)
* [Installation](#installation)
* [Usage](#usage)
* [Grading Methods](#grading-methods)
* [AI Tutor](#ai-tutor)
* [Dependencies](#dependencies)
* [License](#license)

---

## Features

* Upload **Answer Key PDFs** and parse questions automatically.
* Upload multiple **Student Answer PDFs** for automated grading.
* Supports **MCQs** and **descriptive answers**.
* **Semantic similarity scoring** using TF-IDF and AI feedback.
* **Keyword matching** grading for concise evaluation.
* Optional **AI explanations for MCQs**.
* Interactive **results dashboard** with:

  * Total scores
  * Detailed feedback per question
  * Downloadable CSV of results
  * Class performance visualization (Bar, Pie, Box, Histogram, Heatmap)
* Built-in **AI Doubt Solver** to answer student queries instantly.
* Sleek **dark-themed Streamlit UI**.

---

## Installation

1. Clone the repository:

```bash
git clone https://github.com/HARI-45-FAV/AI_EXAM_EVALUTION.git
cd AI_EXAM_EVALUTION
```

2. Create and activate a virtual environment:

```bash
python -m venv venv
# Windows
.\venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Make sure **Ollama** is installed and running (required for AI feedback & tutor):

```bash
ollama start
```

---

## Usage

1. Run the Streamlit app:

```bash
streamlit run app.py
```

2. **Step 1:** Upload the **Answer Key PDF**.
3. **Step 2:** Upload **Student PDFs** and click **Run Grading**.
4. **Step 3:** View results, download CSV, and analyze performance.
5. Ask the **AI Doubt Solver** for any exam-related question.

---

## Grading Methods

* **Semantic Similarity + AI Feedback**
  Uses TF-IDF cosine similarity and AI-generated feedback to grade descriptive answers.

* **Keyword Matching**
  Grades based on matched keywords between student answer and answer key.

* **MCQs**
  Automatically grades multiple-choice questions and optionally provides AI explanations.

---

## AI Tutor

* Ask any exam or topic-related question.
* The AI Tutor responds with concise explanations using the Ollama model.

---

## Dependencies

* Python 3.10+
* [Streamlit](https://streamlit.io/)
* [PyMuPDF (fitz)](https://pymupdf.readthedocs.io/)
* [Ollama Python Client](https://ollama.com/)
* [scikit-learn](https://scikit-learn.org/)
* [pandas](https://pandas.pydata.org/)
* [plotly](https://plotly.com/python/)

---
