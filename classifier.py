#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import re

class TransactionClassifier:
    
    def __init__(self, dataset, transaction_col, category_col, test_percentage, bow_max_features, features_percentile):
        print("Initializing classification model")
        self.classifier = SGDClassifier()
        
        x = dataset.iloc[:, transaction_col].values
        y = dataset.iloc[:, category_col].values
        
        # Step 1: Preprocess
        corpus = self.__preprocess(x)

        # Step 2: Separate dataset into training and test
        (X_train, X_test, y_train, y_test) = self.__separate_train_test(corpus, y, test_percentage)

        # Step 3: Generate Bag of Words (BOW)
        (X_train_bow, X_test_bow) = self.__bow_transformation(X_train, X_test, bow_max_features)

        # Step 4: Extract features
        (X_train_features, X_test_features) = self.__extract_features(X_train_bow, X_test_bow, y_train, features_percentile)

        # Step 5: Train model
        self.__train_model(X_train_features, y_train)

        # Step 6: Test model
        y_pred = self.__test_model(X_test_features, y_test)

        # Step 7: Generate metrics
        self.__calculate_metrics(X_test_features, y_test, y_pred)
        
        print("Finished classification with score: " + str(self.score))
        
    def __preprocess(self, x):
        stemmer = SnowballStemmer("portuguese")
        corpus = []
        for line in x:
            transaction = re.sub('[^a-zA-Z]', ' ', line)
            transaction = transaction.split()
            transaction = [stemmer.stem(word) for word in transaction if len(word) > 1]
            corpus.append(' '.join(transaction))
        return corpus
    
    def __separate_train_test(self, x, y, test_percentage):
        test_size = test_percentage / 100.0
        return train_test_split(x, y, test_size = test_size)
        
    def __bow_transformation(self, X_train, X_test, bow_max_features):
        self.vectorizer = CountVectorizer(max_features = bow_max_features, strip_accents = 'unicode')
        X_train_bow = self.vectorizer.fit_transform(X_train)
        X_test_bow = self.vectorizer.transform(X_test)
        
        return X_train_bow.toarray(), X_test_bow.toarray()
    
    def __extract_features(self, X_train_bow, X_test_bow, y_train, features_percentile):
        self.feature_selector = SelectPercentile(f_classif, percentile = features_percentile)
        X_train_features = self.feature_selector.fit_transform(X_train_bow, y_train)
        X_test_features = self.feature_selector.transform(X_test_bow)
        
        return X_train_features, X_test_features
        
    def __train_model(self, features, y):
        self.classifier.fit(features, y)
    
    def __test_model(self, features, y_test):
        return self.classifier.predict(features)
    
    def __calculate_metrics(self, features, y_test, y_pred):
        self.cm = confusion_matrix(y_test, y_pred)
        self.score = self.classifier.score(features, y_test)
        
    def predict_dataset(self, dataset, transactionColumn):
        x = dataset.iloc[:,transactionColumn].values
        corpus = self.__preprocess(x)
        X_bow = self.vectorizer.transform(corpus).toarray()
        X_features = self.feature_selector.transform(X_bow)
        return self.classifier.predict(X_features)
    
    def predict_transaction(self, transaction):
        corpus = self.__preprocess([transaction])
        X_bow = self.vectorizer.transform(corpus).toarray()
        X_features = self.feature_selector.transform(X_bow)
        return self.classifier.predict(X_features)
	
