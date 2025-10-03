# app.py
import streamlit as st
import re
import pandas as pd
import fitz  # PyMuPDF
from ollama import Client
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px
import plotly.graph_objects as go

# --- Page Configuration ---
st.set_page_config(
    page_title="AutoGrader Pro+",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS ---
def load_css():
    st.markdown("""
        <style>
        .stApp { 
            background: linear-gradient(160deg, #1c1c2e 0%, #26264d 100%);
            color: #e0e0e0; 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        div[data-testid="stVerticalBlock"] > [data-testid="stHorizontalBlock"] > [data-testid="stVerticalBlock"] {
            border: 1px solid #3a3a5e; 
            border-radius: 12px; 
            padding: 30px;
            background-color: #2b2b50; 
            box-shadow: 0 6px 14px rgba(0,0,0,0.35);
        }
        h1 { color: #f5a623; text-align: center; font-weight: 800; text-shadow: 2px 2px 4px #000000; cursor: pointer; transition: all 0.3s; }
        h1:hover { color: #ffc857; transform: scale(1.05); }
        h2 { color: #e0e0e0; border-bottom: 2px solid #f5a623; padding-bottom: 6px; font-weight: 600; }
        .stButton > button { border: 2px solid #f5a623; border-radius: 20px; color: #f5a623; background-color: #1f1f3a; padding: 12px 28px; font-weight: 700; transition: all 0.3s; box-shadow: 0 4px #b38c1d; }
        .stButton > button:hover { background-color: #f5a623; color: #1f1f3a; box-shadow: 0 2px #b38c1d; transform: translateY(2px); }
        .stButton > button:active { box-shadow: 0 1px #b38c1d; transform: translateY(4px); }
        .st-expander { border: 1px solid #3a3a5e !important; border-radius: 12px !important; background-color: #2c2c55 !important; }
        .stMetric { color: #e0e0e0; }
        textarea, input[type=text] { background-color: #333355; color: #e0e0e0; border-radius: 8px; border: 1px solid #3a3a5e; padding: 8px; }
        </style>
    """, unsafe_allow_html=True)
load_css()

# --- Ollama Client ---
try:
    client = Client()
except Exception:
    st.error("Ollama connection failed. Please ensure Ollama is running.")
    st.stop()

# --- Grading Functions ---
def grade_semantic(student_answer, key_answer, max_marks):
    if not student_answer.strip(): 
        return 0, "No answer provided."
    try:
        vectorizer = TfidfVectorizer().fit([student_answer, key_answer])
        vectors = vectorizer.transform([student_answer, key_answer])
        similarity = cosine_similarity(vectors[0], vectors[1])[0][0]
        score = similarity * max_marks
        feedback = f"Semantic similarity: {similarity*100:.1f}%. Score: {round(score,2)}/{max_marks}"
        return round(score,2), feedback
    except Exception as e:
        return 0, f"Error in semantic grading: {e}"

def ai_feedback(model, question, key_answer, student_answer):
    prompt = f"""
    You are a teacher. Evaluate the student's answer based on the key answer.
    Question: "{question}"
    Answer Key: "{key_answer}"
    Student Answer: "{student_answer}"
    Provide one concise feedback sentence.
    """
    try:
        resp = client.chat(model=model, messages=[{'role':'user','content':prompt}])
        return resp['message']['content']
    except:
        return "AI feedback could not be generated."

def generate_mcq_explanation(model, question, correct_option, correct_answer_text):
    prompt = f'Explain why "{correct_option}" is correct for "{question}". Answer: "{correct_answer_text}".'
    try:
        resp = client.chat(model=model, messages=[{'role':'user','content':prompt}])
        return resp['message']['content']
    except:
        return "Could not generate explanation."

def ask_ai_tutor(model, user_question):
    prompt = f"You are a helpful AI tutor. Answer this question concisely:\n{user_question}"
    try:
        resp = client.chat(model=model, messages=[{'role':'user','content':prompt}])
        return resp['message']['content']
    except Exception as e:
        return f"Error: {e}"

def grade_descriptive_by_keyword(student_answer, key_answer, max_marks):
    STOP_WORDS = set(['a','an','and','the','is','it','in','on','of','for','to'])
    key_words = set(w for w in re.findall(r'\b\w+\b', key_answer.lower()) if w not in STOP_WORDS)
    if not key_words: return 0, "No keywords in answer key."
    student_words = set(w for w in re.findall(r'\b\w+\b', student_answer.lower()) if w not in STOP_WORDS)
    matched = key_words.intersection(student_words)
    score = (len(matched)/len(key_words))*max_marks
    feedback = f"Matched {len(matched)}/{len(key_words)} keywords ({(len(matched)/len(key_words))*100:.0f}%)."
    return round(score,2), feedback

# --- PDF Parsing ---
def extract_text_from_pdf_filelike(file_like):
    try:
        file_like.seek(0)
        pdf_bytes = file_like.read()
        with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
            return "\n\n".join([page.get_text() for page in doc])
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return ""

def split_questions(text):
    text = text.replace("\r\n","\n")
    pattern = re.compile(r'(?:^|\n)(?:Q(?:uestion)?\s*\.?\s*)(\d+)[\.\):]?\s*', re.IGNORECASE)
    matches = list(pattern.finditer(text))
    if not matches: return [{"qnum":i+1,"raw":b.strip()} for i,b in enumerate(text.split("\n\n")) if b.strip()]
    parts=[]
    for i,m in enumerate(matches):
        start, qnum = m.end(), int(m.group(1))
        end = matches[i+1].start() if i+1<len(matches) else len(text)
        parts.append({"qnum":qnum, "raw":text[start:end].strip()})
    return parts

def parse_question_block(block_text):
    q = {"type":"descriptive","question":"","choices":{},"answer":None}
    mq = re.search(r'Question[:\s]*(.*)', block_text, flags=re.IGNORECASE)
    q["question"] = mq.group(1).strip() if mq else (block_text.splitlines() or [""])[0]
    choices = re.findall(r'^\s*([A-D])[\.\)]\s*(.+)$', block_text, flags=re.MULTILINE)
    if choices:
        q["type"]="mcq"
        q["choices"]={letter.upper():text.strip() for letter,text in choices}
    m_ans = re.search(r'(Correct Answer|Answer)[:\s]*\(\s*([A-D])\s*\)', block_text, flags=re.IGNORECASE|re.MULTILINE)
    if m_ans:
        q["answer"] = m_ans.group(2).strip().upper()
        q["type"]="mcq"
    else:
        m_ans_desc = re.search(r'(Correct Answer|Answer)[:\s]*(.+)$', block_text, flags=re.IGNORECASE|re.MULTILINE)
        if m_ans_desc: q["answer"] = m_ans_desc.group(2).strip()
    return q

# --- Sidebar Settings ---
with st.sidebar:
    st.markdown("<h2>AutoGrader Pro+ 🧠</h2>", unsafe_allow_html=True)
    st.info("High-Accuracy AI Grading Assistant")
    st.markdown("<h3>Settings</h3>", unsafe_allow_html=True)
    st.session_state.ollama_model = st.text_input("Ollama Model", value="gemma:2b")
    st.session_state.grading_method = st.selectbox("Descriptive Grading Method", ["Semantic Similarity + AI Feedback", "Keyword Matching"])
    st.session_state.generate_explanations = st.checkbox("Generate AI Explanations for MCQs")

# --- Main App: Dashboard Toggle ---
if 'show_dashboard' not in st.session_state:
    st.session_state.show_dashboard = False

if st.button("🧠 AutoGrader Pro+: Exam Evaluator"):
    st.session_state.show_dashboard = not st.session_state.show_dashboard
    st.balloons()

st.markdown("<hr>", unsafe_allow_html=True)

# --- Step 1: Upload Answer Key ---
with st.container():
    st.markdown("<h2>Step 1: Upload Answer Key</h2>", unsafe_allow_html=True)
    uploaded_key = st.file_uploader("Upload Answer Key PDF", type=["pdf"], key="key_uploader")
    if uploaded_key:
        raw_text = extract_text_from_pdf_filelike(uploaded_key)
        if raw_text.strip():
            st.session_state.parsed_questions = [{**parse_question_block(p["raw"]),"qnum":p["qnum"]} for p in split_questions(raw_text)]
    if st.session_state.get('parsed_questions'):
        st.success(f"✅ Parsed {len(st.session_state.parsed_questions)} questions. Set marks below.")
        for q in st.session_state.parsed_questions:
            with st.expander(f"Q{q['qnum']}: {q['question'][:50]}... ({q['type']})"):
                default = 1 if q['type']=="mcq" else 5
                q['max_marks'] = st.number_input("Max Marks", min_value=1, value=q.get('max_marks',default), key=f"q_{q['qnum']}_max")
                q['question'] = st.text_area("Question Text", value=q["question"], key=f"q_{q['qnum']}_q")
                q['answer'] = st.text_input("Correct Answer", value=q.get("answer",""), key=f"q_{q['qnum']}_ans")

# --- Step 2: Upload Student Answers & Grade ---
with st.container():
    st.markdown("<h2>Step 2: Upload Student Answers & Grade</h2>", unsafe_allow_html=True)
    student_files = st.file_uploader("Upload Student PDFs", type=["pdf"], accept_multiple_files=True, key="student_uploader")
    
    if st.button("🚀 Run Grading") and st.session_state.get('parsed_questions') and student_files:
        results=[]
        progress = st.progress(0, text="Starting grading...")
        for i, sf in enumerate(student_files):
            student_name = sf.name.rsplit(".",1)[0]
            progress.progress(i/len(student_files), text=f"Grading {student_name}...")
            student_text = extract_text_from_pdf_filelike(sf)
            if not student_text.strip():
                st.warning(f"No text found in {sf.name}. Skipping.")
                continue
            record={"student":student_name,"total_score":0.0,"detail":[]}
            for q in st.session_state.parsed_questions:
                pat = re.compile(r'(?:Q(?:uestion)?\s*'+str(q['qnum'])+r'[\.\):]?\s*)(.*?)(?=(?:Q(?:uestion)?\s*\d+[\.\):]|\Z))', re.IGNORECASE|re.S)
                match = pat.search(student_text)
                student_block = match.group(1).strip() if match else student_text

                if q["type"]=="mcq":
                    mcq_pat = re.search(r'\(\s*([A-D])\s*\)', student_block, re.IGNORECASE)
                    s_ans = mcq_pat.group(1).upper() if mcq_pat else "Not Found"
                    is_correct = s_ans==q["answer"]
                    score = q['max_marks'] if is_correct else 0
                    feedback = "Correct" if is_correct else f"Incorrect. Correct: ({q['answer']})"
                    if is_correct and st.session_state.generate_explanations:
                        feedback += " | Explanation: " + generate_mcq_explanation(st.session_state.ollama_model, q['question'], q['answer'], q['choices'].get(q['answer'], ''))
                else:
                    s_ans = student_block or "No answer found."
                    if st.session_state.grading_method=="Semantic Similarity + AI Feedback":
                        score, sem_feedback = grade_semantic(s_ans, q['answer'], q['max_marks'])
                        feedback = sem_feedback + "\nAI Feedback: " + ai_feedback(st.session_state.ollama_model, q['question'], q['answer'], s_ans)
                    else:
                        score, feedback = grade_descriptive_by_keyword(s_ans, q['answer'], q['max_marks'])

                record[f"Q{q['qnum']}_score"]=score
                record["total_score"]+=score
                record["detail"].append({"qnum":q['qnum'], "student_answer":s_ans, "score":score, "feedback":feedback, "type":q['type']})
            results.append(record)
        progress.progress(1.0, text="Grading complete!")
        st.session_state.results=results

# --- Step 3: Display Results + Analysis ---
if st.session_state.get('results'):
    with st.container():
        st.markdown("<h2>Step 3: Grading Results</h2>", unsafe_allow_html=True)
        results = st.session_state.results
        avg_score = sum(r['total_score'] for r in results)/len(results) if results else 0
        col1,col2,_=st.columns(3)
        col1.metric("Students Graded", f"{len(results)} 🧑‍🎓")
        col2.metric("Average Score", f"{avg_score:.2f} 📊")

        # DataFrame
        df_rows=[{"student":r["student"],"total_score":round(r["total_score"],2), **{f"Q{q['qnum']}_score": r.get(f"Q{q['qnum']}_score",0) for q in st.session_state.parsed_questions}} for r in results]
        df=pd.DataFrame(df_rows)
        st.dataframe(df,use_container_width=True)
        st.download_button("📥 Download CSV", df.to_csv(index=False).encode('utf-8'), "exam_grades.csv", "text/csv")

        # Detailed Feedback
        st.markdown("<h3>Detailed Feedback per Student</h3>", unsafe_allow_html=True)
        for r in results:
            with st.expander(f"{r['student']} (Total: {r['total_score']:.2f})"):
                for d in r['detail']:
                    st.markdown(f"**Question {d['qnum']} ({d['type']})**")
                    st.text_area("Student's Answer", d['student_answer'], height=100, disabled=True, key=f"detail_{r['student']}_{d['qnum']}")
                    st.info(f"**Score:** {d['score']:.2f} | **Feedback:** {d['feedback']}")

        # --- Visualization ---
        st.markdown("<h3>📊 Class Performance Analysis</h3>", unsafe_allow_html=True)
        chart_type = st.selectbox("Select Chart Type", ["Bar Chart", "Pie Chart", "Box Plot", "Histogram"])
        metric_option = st.selectbox("Select Metric", ["total_score"] + [f"Q{q['qnum']}_score" for q in st.session_state.parsed_questions])

        if chart_type=="Bar Chart":
            fig = px.bar(df, x="student", y=metric_option, text=metric_option, color=metric_option)
            st.plotly_chart(fig, use_container_width=True)
        elif chart_type=="Pie Chart":
            fig = px.pie(df, names="student", values=metric_option)
            st.plotly_chart(fig, use_container_width=True)
        elif chart_type=="Box Plot":
            fig = px.box(df, y=metric_option, points="all")
            st.plotly_chart(fig, use_container_width=True)
        elif chart_type=="Histogram":
            fig = px.histogram(df, x=metric_option, nbins=10)
            st.plotly_chart(fig, use_container_width=True)

        # Question-wise Heatmap
        heatmap_data = df[[f"Q{q['qnum']}_score" for q in st.session_state.parsed_questions]]
        fig = go.Figure(data=go.Heatmap(
            z=heatmap_data.values,
            x=[f"Q{q['qnum']}" for q in st.session_state.parsed_questions],
            y=df['student'],
            colorscale='Viridis',
            colorbar=dict(title="Score")
        ))
        fig.update_layout(title="Question-wise Student Performance Heatmap", xaxis_title="Questions", yaxis_title="Students")
        st.plotly_chart(fig, use_container_width=True)

# --- AI Doubt Solver ---
with st.container():
    st.markdown("---")
    st.markdown("<h2>🤖 AI Doubt Solver</h2>", unsafe_allow_html=True)
    st.info("Ask any question about the exam or topic.")
    user_doubt = st.text_input("Enter your question:", key="doubt_input", placeholder="e.g., Explain photosynthesis simply")
    if st.button("💬 Ask AI Tutor"):
        if user_doubt:
            with st.spinner("AI Tutor is thinking..."):
                ai_answer = ask_ai_tutor(st.session_state.ollama_model, user_doubt)
                st.session_state.ai_answer = ai_answer
        else:
            st.warning("Enter a question first.")
    if 'ai_answer' in st.session_state and st.session_state.ai_answer:
        st.success("**AI Tutor's Answer:**")
        st.markdown(st.session_state.ai_answer)
