"""Week 7 - Hugging Face Pipelines. Run with: py pipelines.py"""
from pathlib import Path
import pandas as pd
from transformers import pipeline

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "dataset"
source = pd.read_csv(DATA / "input_text.csv")
text = str(source.loc[0, "text"])

def make_pipeline(task, model=None):
    print(f"\nLoading {task}...")
    return pipeline(task, model=model) if model else pipeline(task)

# 1. Sentiment analysis
sentiment = make_pipeline("sentiment-analysis")
sentiment_result = sentiment(text[:2000])[0]
pd.DataFrame([{
    "id": 1, "text": text[:2000], "label": sentiment_result["label"],
    "score": round(float(sentiment_result["score"]), 4), "status": "generated"
}]).to_csv(DATA / "sentiment_results.csv", index=False)

# 2. Summarization; compare two max_length values
summarizer = make_pipeline("summarization", "sshleifer/distilbart-cnn-12-6")
summary_rows = []
for max_len, min_len in [(70, 20), (120, 35)]:
    result = summarizer(text[:4000], max_length=max_len, min_length=min_len, do_sample=False)[0]
    summary_rows.append({
        "id": 1, "original_text": text, "summary": result["summary_text"],
        "max_length": max_len, "status": "generated"
    })
summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(DATA / "summarized_text.csv", index=False)

# 3. Translation from English to French
translator = make_pipeline("translation_en_to_fr", "Helsinki-NLP/opus-mt-en-fr")
translation = translator(text[:800])[0]["translation_text"]
pd.DataFrame([{
    "id": 1, "source_language": "English", "target_language": "French",
    "original_text": text[:800], "translation": translation, "status": "generated"
}]).to_csv(DATA / "translated_text.csv", index=False)

# 4. Question answering: five questions from the article
qa = make_pipeline("question-answering", "distilbert/distilbert-base-cased-distilled-squad")
questions = [
    "What does sentiment analysis classify?",
    "Why should people verify AI-generated information?",
    "What does summarization help readers do?",
    "Which library provides the pipeline interface?",
    "Why can translation require expert checking?"
]
qa_rows = []
for question in questions:
    answer = qa(question=question, context=text)
    qa_rows.append({
        "question": question, "context": text, "answer": answer["answer"],
        "score": round(float(answer["score"]), 4), "status": "generated"
    })
qa_df = pd.DataFrame(qa_rows)
qa_df.to_csv(DATA / "qa_results.csv", index=False)

# 5. Comparison CSV
comparison = []
for _, row in summary_df.iterrows():
    comparison.append({"task": "Summarization", "input": text[:250], "output": row["summary"],
                       "observation": f"max_length={row['max_length']}", "status": "generated"})
comparison.append({"task": "Translation", "input": text[:250], "output": translation,
                   "observation": "English translated into French", "status": "generated"})
comparison.append({"task": "Sentiment analysis", "input": text[:250], "output": sentiment_result["label"],
                   "observation": f"confidence={sentiment_result['score']:.4f}", "status": "generated"})
for row in qa_rows:
    comparison.append({"task": "Question answering", "input": row["question"], "output": row["answer"],
                       "observation": f"confidence={row['score']:.4f}", "status": "generated"})
pd.DataFrame(comparison).to_csv(DATA / "pipeline_comparison.csv", index=False)

print("\nCompleted. Generated CSV files are in the dataset folder.")
