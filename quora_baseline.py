import pandas as pd
from nltk.tokenize.toktok import ToktokTokenizer
import nltk,re
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

train_data_path = r'input_data\train.csv'
test_data_path = r'input_data\test.csv'

train_df = pd.read_csv(train_data_path)
test_df = pd.read_csv(test_data_path)

tokenizer = ToktokTokenizer()
stop_words = set(stopwords.words('english'))


target_value_0 = train_df['question_text'][train_df['target'] == 0]
target_value_1 = train_df['question_text'][train_df['target'] == 1]




def Punctuation(string):
    # punctuation marks
    punctuations = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
    for x in string.lower():
        if x in punctuations:
            string = string.replace(x, "")
    return string

def remove_digits(text):
    text = re.sub(r'\d +', '', text)
    return text

def simple_stemmer(text):
    text = text.lower()
    ps = nltk.porter.PorterStemmer()
    text = ' '.join([ps.stem(word) for word in text.split()])
    return text


def remove_stopwords(text, is_lower_case=False):
    tokens = tokenizer.tokenize(text)
    tokens = [token.strip() for token in tokens]
    if is_lower_case:
        filtered_tokens = [token for token in tokens if token not in stop_words]
    else:
        filtered_tokens = [token for token in tokens if token.lower() not in stop_words]
    filtered_text = ' '.join(filtered_tokens)
    return filtered_text


# def token(text):
#     token = tokenizer.tokenize(text)
#     return token

target = train_df['target'][:100]
X = train_df['question_text'][:100]
X_test = test_df['question_text']

# train and evalution data preprocessing
no_punct = [Punctuation(i)for i in X]
rm_d = [remove_digits(i) for i in no_punct]
stem = [simple_stemmer(i) for i in rm_d]
r_sw = [remove_stopwords(i) for i in stem]

# test_data preprocessing
no_punct_t = [Punctuation(i)for i in X_test]
stem_t = [simple_stemmer(i) for i in no_punct]
r_sw_t = [remove_stopwords(i) for i in stem]


# convert into tfidf
tfidf = TfidfVectorizer(use_idf=True)
words_fit = tfidf.fit(r_sw)
word_corpus = tfidf.transform(r_sw)

# test data transformation
word_corpus_test = tfidf.transform(r_sw_t)

# test_train_split

x_train, x_test, y_train, y_test = train_test_split(word_corpus,target,test_size=0.3)
# clf

clf = RandomForestClassifier(n_estimators=1750, min_samples_split=7,min_samples_leaf=7)
clf.fit(x_train,y_train)

# saving clf model
joblib.dump(clf, 'random_forest_quora.pkl')

# loading saved model
# loaded_model = joblib.load('random_forest_quora.pkl')
# loaded_model.predict(x_train)

train_preds = clf.predict(x_train)
eval_preds = clf.predict(x_test)

# train and evalution metrics
print(classification_report(train_preds,y_train))
print(classification_report(eval_preds,y_test))

# test data preds

# test_preds = clf.predict(word_corpus_test)

# sample_submission = pd.DataFrame(columns=['qid', 'prediction'])
# sample_submission['qid'] = test_df['qid']
# sample_submission['prediction'] = test_preds
# sample_submission.to_csv('sample_submission.csv')
