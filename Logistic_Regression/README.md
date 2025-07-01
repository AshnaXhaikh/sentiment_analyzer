# 🧠 Sentiment Analyzer

🚀 **[My Live App on Hugging Face Spaces](https://huggingface.co/spaces/ashnaxhaikh/Sentiment_Analysis)**  
Try the sentiment classifier in your browser with just one click!

This is a simple and lightweight **Sentiment Analysis web app** built with:
- ✅ Logistic Regression
- ✅ TF-IDF Vectorizer
- ✅ Trained on Sentiment140 dataset
- ✅ Deployed using Gradio on Hugging Face Spaces


---

## 📊 Dataset

We used the **[Sentiment140 dataset](https://www.kaggle.com/datasets/kazanova/sentiment140)**, which contains 1.6 million tweets labeled as:

- `0` = Negative sentiment
- `4` = Positive sentiment

We mapped the labels to binary form:
- `0 → 0` (Negative)
- `4 → 1` (Positive)

---

## NLP techniques used

1. **Stopwords Removal**
Stopwords are common words (like "the", "is", "and") that usually do not carry important meaning in sentiment classification.

2. Stemming and Lemmatization
🔸 Stemming
Stemming reduces words to their root form by chopping off suffixes.

Example: "happily", "happiness" → "happi"

🔸 **Lemmatization** (Preferred in this project)
Lemmatization converts words to their dictionary base form (lemma) using vocabulary and grammar rules.

Example: "running", "ran" → "run"

We used lemmatization over stemming because:
- It improves the quality of TF-IDF features
- Better suited for tasks where word meaning matters (like sentiment)

3. **TF-IDF Vectorization**
TF-IDF (Term Frequency-Inverse Document Frequency) converts text into numerical vectors based on word importance:

TF: Frequency of a word in a sentence
IDF: Rarity of a word across all documents

---

## 🧪 Model

We used **Logistic Regression**, Support Vector Machine (SVM), and Random Forest classifiers, xgboost, but Logistic Regression performed best.
