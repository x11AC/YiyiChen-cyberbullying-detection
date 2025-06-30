import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from tpot import TPOTClassifier
from tqdm import tqdm
import logging
import time

# Set up logging with UTF-8 encoding
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training_log.txt', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger()

# Download NLTK resources
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# Cleaning function
def clean_text(text):
    """Clean text data for cyberbullying detection."""
    if pd.isna(text) or not text:
        return ""
    text = str(text)
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\d', ' ', text)
    text = text.lower()
    stop_words = set(stopwords.words('english'))
    text = ' '.join(word for word in text.split() if word not in stop_words)
    lemmatizer = WordNetLemmatizer()
    text = ' '.join(lemmatizer.lemmatize(word) for word in text.split())
    return text

# Load local data
logger.info("Loading data from local file")
file_path = 'D:/NYUSH/curaJOY/raw_data.txt'
try:
    data = pd.read_csv(file_path, sep='\t', header=None, names=['text'])
except Exception as e:
    logger.error(f"Error loading data: {e}")
    raise

logger.info(f"Loaded {len(data)} records")
logger.info(f"Raw data sample:\n{data.head(10).to_string()}")

# Parse text and label using string splitting
def parse_text_label(row):
    try:
        text = row['text'].strip()
        if ': True' in text:
            parts = text.rsplit(': True', 1)
            return pd.Series({'text': parts[0].strip("' "), 'label': True})
        elif ': False' in text:
            parts = text.rsplit(': False', 1)
            return pd.Series({'text': parts[0].strip("' "), 'label': False})
        else:
            logger.warning(f"Invalid format in row: {text}")
            return pd.Series({'text': '', 'label': None})
    except Exception as e:
        logger.warning(f"Error parsing row: {text}, error: {e}")
        return pd.Series({'text': '', 'label': None})

logger.info("Parsing text and label")
data = data.apply(parse_text_label, axis=1)
logger.info(f"Parsed data sample:\n{data.head(10).to_string()}")
logger.info(f"Missing values:\n{data.isnull().sum().to_string()}")

# Clean the text with progress bar
logger.info("Starting text cleaning")
data['text'] = [clean_text(text) for text in tqdm(data['text'], desc="Cleaning Text")]
logger.info("Text cleaning completed")

# Handle missing or invalid data
data = data[data['text'].str.strip() != '']  # Remove empty text
data = data.dropna(subset=['label'])  # Drop rows with missing labels
data['label'] = data['label'].astype(bool)
logger.info(f"After cleaning, {len(data)} records remain")
if len(data) < 10:
    logger.error(f"Too few records ({len(data)}) remain. Need at least 10 for train/test split.")
    raise ValueError("Too few records after cleaning")

# Split the data into train/test sets
logger.info("Splitting data into train/test sets")
X_train, X_test, y_train, y_test = train_test_split(
    data['text'], data['label'], test_size=0.2, random_state=42
)
logger.info(f"Training set size: {len(X_train)}, Test set size: {len(X_test)}")

# Feature extraction with TF-IDF
logger.info("Extracting TF-IDF features")
vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train).toarray()
X_test_vec = vectorizer.transform(X_test).toarray()
logger.info("Feature extraction completed")

# Train baseline SVM model with RBF kernel
logger.info("Starting baseline SVM training (RBF kernel)")
start_time = time.time()
svm_model = SVC(kernel='rbf', probability=True, random_state=42, C=1.0, gamma='scale', verbose=True, class_weight='balanced')
svm_model.fit(X_train_vec, y_train)
logger.info(f"Baseline SVM training completed in {time.time() - start_time:.2f} seconds")

# Evaluate baseline SVM
y_pred_svm = svm_model.predict(X_test_vec)
logger.info("Baseline SVM (RBF Kernel) Classification Report:")
print("\nBaseline SVM (RBF Kernel) Classification Report:")
print(classification_report(y_test, y_pred_svm, target_names=['Non-Cyberbullying', 'Cyberbullying']))
logger.info("Baseline SVM (RBF Kernel) Confusion Matrix:")
print("Baseline SVM (RBF Kernel) Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_svm))

# TPOT optimization with progress tracking
logger.info("Starting TPOT optimization")
start_time = time.time()
tpot = TPOTClassifier(generations=5, population_size=20, verbosity=2, random_state=42, max_time_mins=3, max_eval_time_mins=0.5)
tpot.fit(X_train_vec, y_train)
logger.info(f"TPOT optimization completed in {time.time() - start_time:.2f} seconds")

# Evaluate TPOT-optimized model
y_pred_tpot = tpot.predict(X_test_vec)
logger.info("TPOT-Optimized Model Classification Report:")
print("\nTPOT-Optimized Model Classification Report:")
print(classification_report(y_test, y_pred_tpot, target_names=['Non-Cyberbullying', 'Cyberbullying']))
logger.info("TPOT-Optimized Model Confusion Matrix:")
print("TPOT-Optimized Model Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_tpot))

# Export the TPOT pipeline
tpot.export('tpot_cyberbullying_pipeline_rbf.py')
logger.info("TPOT pipeline exported to 'tpot_cyberbullying_pipeline_rbf.py'")