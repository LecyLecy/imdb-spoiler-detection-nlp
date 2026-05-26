# IMDb Spoiler Detection NLP

This project builds a text classification system to detect whether an IMDb review contains spoilers. The model uses a classic Natural Language Processing (NLP) approach based on TF-IDF, an additional review-length feature, and several lightweight machine learning algorithms without deep learning. This project also provides a simple Streamlit app for trying spoiler predictions interactively.

Try the app:

```text
https://imdb-spoiler-detection-nlp.streamlit.app/
```

## Overview

The system classifies reviews into two categories:

- `Spoiler`
- `Non-Spoiler`

The main goal of this project is to help automatically filter reviews so users can read movie or series reviews without being exposed to story details, plot twists, or endings.

## Dataset

The dataset used is the IMDb Spoiler Dataset from Kaggle:

https://www.kaggle.com/datasets/rmisra/imdb-spoiler-dataset

Main file:

```text
IMDB_reviews.json
```

Main columns used:

| Column          | Description                                       |
| --------------- | ------------------------------------------------- |
| `review_text` | Full IMDb review text                             |
| `is_spoiler`  | Target label, 1 for spoiler and 0 for non-spoiler |

This project uses the full dataset without manual balancing so the data distribution still represents the original conditions.

## Project Workflow

Main notebook workflow:

1. Load the dataset from Kaggle or local cache
2. Perform Exploratory Data Analysis (EDA)
3. Perform preprocessing and feature engineering
4. Split the data with an 80% train and 20% test ratio
5. Extract features using TF-IDF unigrams and bigrams
6. Train several machine learning models
7. Evaluate models using Accuracy, Precision, Recall, F1-Score, Confusion Matrix, and Classification Report
8. Select the best model based on F1-Score
9. Test 20 random review samples
10. Save the best model pipeline

## Exploratory Data Analysis

EDA is performed to understand the dataset characteristics before model training:

- Distribution of spoiler and non-spoiler labels
- Class proportions to inspect data imbalance
- Review length distribution based on `word_count`
- Top words in spoiler and non-spoiler reviews

The model evaluation visualization is saved at:

```text
figures/best_model_visualization.png
```

## Preprocessing & Feature Engineering

Preprocessing follows the approach used in the `NLPB.ipynb` notebook, which keeps the original review text and performs transformations inside the pipeline.

Features used:

- `review_text`: raw review text
- `word_count`: number of words in the review as a meta-feature

Feature extraction:

- `TfidfVectorizer`
- `ngram_range=(1, 2)` for unigrams and bigrams
- `max_features=15000`
- Custom English stop words, including common stop words and several movie-domain words such as `movie`, `film`, `character`, and `story`
- `StandardScaler` to standardize the numeric `word_count` feature

This approach keeps sentence context available while reducing noise from overly common words.

## Models

The notebook compares several machine learning models:

- SGD Classifier + Meta Feature
- Linear SVC + Meta Feature
- Passive Aggressive Classifier + Meta Feature
- Multinomial Naive Bayes
- Complement Naive Bayes

Logistic Regression is not used in the latest version. The best model is selected based on the highest F1-Score on the test set.

## Evaluation Metrics

Evaluation metrics:

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix
- Classification Report

F1-Score is used as the main metric because spoiler detection requires a balance between finding spoiler reviews and avoiding too many false positives.

## Saved Model

The best model is saved as a complete pipeline so preprocessing, feature extraction, and the classifier can be reused in a single object.

Model outputs:

```text
models/spoiler_detection_pipeline.pkl
models/best_model_name.pkl
models/model_comparison_results.pkl
```

The `app.py` file loads `spoiler_detection_pipeline.pkl` and `best_model_name.pkl` from the `models/` folder. User input is converted into a DataFrame with the `review_text` and `word_count` columns, then predicted using the same pipeline produced by the training notebook.

## Streamlit App

This project includes an interactive web app built with Streamlit:

```text
app.py
```

The app performs the following steps:

- Loads the best model pipeline from the `models/` folder
- Calculates `word_count` from the review entered by the user
- Creates input using the `review_text` and `word_count` column format
- Retrieves the spoiler probability using `predict_proba`
- Displays the `SPOILER DETECTED` or `SAFE TO READ` label

Run the app with:

```bash
streamlit run app.py
```

Make sure the following model files are available before running the app:

```text
models/spoiler_detection_pipeline.pkl
models/best_model_name.pkl
```

If the app fails to load the model because of dependency version differences, rerun the notebook in the active environment to regenerate the `.pkl` files.

## Deployment

The recommended deployment platform is Streamlit Community Cloud because this project already includes `app.py`, `requirements.txt`, and model files in the `models/` folder.

Deployment steps:

1. Make sure the important files are already in the repository:

```text
app.py
requirements.txt
models/spoiler_detection_pipeline.pkl
models/best_model_name.pkl
```

2. Make sure all changes have been pushed to GitHub:

```bash
git status
git add README.md requirements.txt app.py models/
git commit -m "Prepare Streamlit deployment"
git push origin main
```

3. Open Streamlit Community Cloud:

```text
https://share.streamlit.io
```

4. Log in using the GitHub account that has access to this repository.
5. Click `Create app`, then choose the option to deploy an app from an existing repository.
6. Fill in the deployment configuration:

```text
Repository : LecyLecy/imdb-spoiler-detection-nlp
Branch     : main
Main file  : app.py
App URL    : free to choose, depending on the available name
```

7. If prompted to choose a Python version, use Python 3.11 to match the project's development environment.
8. Click deploy and wait for the dependency installation process from `requirements.txt` to finish.
9. After deployment succeeds, copy the Streamlit app link and place it in the `Try the app` placeholder at the top of the README.
10. If deployment fails, check the logs section in Streamlit Cloud. The most likely errors are dependency mismatch or missing model files.

## Repository Structure

```text
imdb-spoiler-detection-nlp/
|-- data/                         # Local dataset folder, ignored by Git
|-- figures/                      # Saved visualizations
|-- models/                       # Saved trained model pipeline
|-- imdb_venv/                    # Local virtual environment, ignored by Git
|-- .env                          # Kaggle credentials, ignored by Git
|-- .env.example                  # Environment variable template
|-- .gitignore
|-- app.py                        # Streamlit app for spoiler prediction
|-- imdb-spoiler-detection-old.ipynb
|-- README.md
|-- requirements.txt
```

## Setup

### 1. Clone Repository

```bash
git clone https://github.com/LecyLecy/imdb-spoiler-detection-nlp.git
cd imdb-spoiler-detection-nlp
```

### 2. Create Python Virtual Environment

```bash
py -3.11 -m venv imdb_venv
imdb_venv\Scripts\activate
```

### 3. Install Dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 4. Register Jupyter Kernel

```bash
python -m ipykernel install --user --name imdb_spoiler_nlp --display-name "Python (imdb_spoiler_nlp)"
```

After that, open the notebook and select the kernel:

```text
Python (imdb_spoiler_nlp)
```

## Kaggle API Setup

Create a `.env` file in the project root:

```env
KAGGLE_USERNAME=your_kaggle_username
KAGGLE_KEY=your_kaggle_api_key
```

The Kaggle API key can be created from:

```text
Kaggle Account Settings -> API -> Create New Token
```

Make sure `.env` is not committed to GitHub.

## Running the Notebook

Open the notebook:

```text
imdb-spoiler-detection-old.ipynb
```

Then run the cells sequentially. The notebook will:

- Check the local dataset
- Download the dataset from Kaggle if it is not available yet
- Load the full dataset
- Perform EDA and visualization
- Create TF-IDF and `word_count` features
- Train several models
- Select the best model based on F1-Score
- Test 20 random review samples
- Save the best model pipeline

After the model is saved, the Streamlit app can be run with:

```bash
streamlit run app.py
```

## Example Prediction

```text
sample_review = The movie ends with the main character dying in the final scene.
The model is 82.89% confident that sample_review is a spoiler
The model is 17.11% confident that sample_review is not a spoiler
Final prediction = spoiler
```

## Git Ignore Recommendation

The following files and folders do not need to be pushed to GitHub:

```gitignore
.env
data/
imdb_venv/
__pycache__/
.ipynb_checkpoints/
*.zip
```

## Notes

The dataset is not included in the repository because the file size is large and it can be downloaded again from Kaggle. The local cache can also be regenerated by running the notebook.

## License

This project uses the IMDb Spoiler Dataset from Kaggle. Please refer to the original dataset page for license details and usage terms.
