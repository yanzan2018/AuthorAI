# -*- coding: utf-8 -*-
from sklearn import datasets  
rawData = datasets.load_files("Literatures")

from sklearn.cross_validation import train_test_split  
book_terms_train, book_terms_test, y_train, y_test = train_test_split(rawData.data, rawData.target, test_size = 0.2)

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(book_terms_train)

tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)   
from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB().fit(X_train_tfidf, y_train)


books_new = book_terms_test
X_new_counts = count_vect.transform(books_new)
X_new_tfidf = tfidf_transformer.transform(X_new_counts)
predicted = clf.predict(X_new_tfidf)


from sklearn.metrics import classification_report
target_names = ['Charles Dickens', 'Mark Twain', 'William Shakespeare']
print(classification_report(y_test, predicted, target_names=target_names))