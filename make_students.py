# make_students.py
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

def make_pdf(filename, student_name, answers):
    c = canvas.Canvas(filename, pagesize=letter)
    c.setFont("Helvetica", 12)
    c.drawString(100, 750, f"Student: {student_name}")

    y = 700
    for i, ans in enumerate(answers, start=1):
        c.drawString(100, y, f"Q{i}. {ans}")
        y -= 50

    c.save()

# --------------------------
# Example Students
# --------------------------

# Suppose Answer Key is:
# Q1 (MCQ): Correct A
# Q2 (Descriptive): Photosynthesis...
# Q3 (MCQ): Correct C

students = {
    "Alice": ["Answer: A", "Plants use sunlight to make food with carbon dioxide.", "Answer: C"],
    "Bob":   ["Answer: B", "Plants eat soil (wrong idea).", "Answer: C"],
    "Charlie":["Answer: A", "They use sunlight to produce sugar and oxygen.", "Answer: D"],
    "Diana": ["Answer: A", "Photosynthesis is the process using sunlight, CO2, water to produce glucose and oxygen.", "Answer: C"],
}

for name, answers in students.items():
    make_pdf(f"{name}.pdf", name, answers)

print("✅ Student PDFs created!")
