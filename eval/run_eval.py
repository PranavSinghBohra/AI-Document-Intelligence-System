import csv
import json
from app.qa import answer_question


def load_questions():
    with open("eval/questions.json", encoding="utf-8") as questions_file:
        return json.load(questions_file)



def format_sources(citations):
    if len(citations) == 0:
        return "none"

    parts = []
    for citation in citations:
        file_name = citation["file_name"]
        page_number = citation["page_number"]
        parts.append(file_name + " p." + str(page_number))
    return "; ".join(parts)


def save_results(rows):
    with open("eval/results.csv", "w", newline="", encoding="utf-8") as results_file:
        writer = csv.writer(results_file)
        writer.writerow(["question", "expected_answer", "retrieved_answer", "source", "correct"])
        writer.writerows(rows)

def run():
    questions = load_questions()

    rows = []
    for item in questions:
        result = answer_question(item["question"])
        sources = format_sources(result["citations"])
        row = [item["question"], item["expected"], result["answer"], sources, ""]
        rows.append(row)

    save_results(rows)




if __name__ == "__main__":
    run()