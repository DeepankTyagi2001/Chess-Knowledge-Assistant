import re

def load_questions(filepath):
    questions = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # matches "1. question text" or "1) question text"
            match = re.match(r"^\d+[\.\)]\s*(.+)", line)
            if match:
                questions.append(match.group(1))
            else:
                questions.append(line)  # fallback if no number prefix
    return questions

