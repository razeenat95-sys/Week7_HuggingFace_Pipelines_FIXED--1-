# Week 7 – Hugging Face Pipelines

**Task Title:** Performing NLP Tasks using Hugging Face Pipelines

## Objective
Use pretrained Transformer pipelines for sentiment analysis, summarization, translation, and question answering, then save and compare the outputs in CSV files.

## Folder structure
```text
HuggingFace_Pipelines_Week7/
├── dataset/
│   ├── input_text.csv
│   ├── summarized_text.csv
│   ├── translated_text.csv
│   ├── sentiment_results.csv
│   ├── qa_results.csv
│   └── pipeline_comparison.csv
├── notebook/
│   └── Week7_HuggingFace_Pipelines.ipynb
├── pipelines.py
├── README.md
└── requirements.txt
```

## Run in VS Code (Windows PowerShell)
1. Extract the ZIP.
2. Open the extracted `HuggingFace_Pipelines_Week7` folder in VS Code.
3. Open Terminal → New Terminal and run:
   ```powershell
   py -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements.txt
   ```
4. Run the project:
   ```powershell
   .venv\Scripts\python.exe pipelines.py
   ```
5. The first run downloads pretrained models. Internet access is required and it may take several minutes.
6. Open the generated CSV files in the `dataset` folder to review the results.

## Notebook
Open `notebook/Week7_HuggingFace_Pipelines.ipynb`, select your Python kernel, and run all cells from top to bottom. The notebook performs the same tasks and writes outputs to the CSV files.

## Home activity covered
- Uses a sample article of approximately 500–800 words.
- Performs sentiment analysis and summarization.
- Compares summaries using different `max_length` values.
- Translates English text into French.
- Answers five questions from the article.
- Saves task outputs and a combined comparison table.

## Notes
- The CSV result files initially contain only headers. Run the script or notebook to generate model outputs.
- Model downloads may fail on a slow connection or if the computer has limited memory.
- Model output may vary and should be checked for factual accuracy.
