import pickle
import random
import argparse
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from tqdm import tqdm
from typing import Tuple
from scipy import sparse as sp
from termcolor import colored
from collections import defaultdict  

def puts(text):
    print(colored(text, "yellow"))

class Vectorizer:
    """
    A vectorizer class that converts text data into a sparse matrix
    """
    def __init__(self, max_vocab_len=10_000) -> None:
        """
        Initialize the vectorizer
        """
        self.vocab = None
        # TODO: Add more class variables if needed

    def fit(self, X_train: np.ndarray) -> None:
        """
        Fit the vectorizer on the training data
        :param X_train: np.ndarray
            The training sentences
        :return: None
        """
        # TODO: count the occurrences of each word
        print("AKASH => ", "X-train size:", X_train.shape)

        total_unique_words = 0
        word_to_count = {}
        word_to_index = {}
        index = 0
        for sentence in X_train:
            for word in sentence.split(" "):
                if word in word_to_count:
                    word_to_count[word] += 1
                else:
                    word_to_count[word] = 1
                    word_to_index[word] = index
                    index += 1
                    total_unique_words += 1


        assert len(word_to_count) == total_unique_words
        print("AKASH => ", "Total Unique Words:" ,total_unique_words)

        # get a few words
        # puts(word_to_count["back"])
        # top_words_to_count = sorted(word_to_count.items(), key = lambda item: item[1], reverse=True)
        # for k, v in top_words_to_count[:10000]:
        #     print(k, v)

                    


        # TODO: sort the words based on frequency
        top_words_to_count = sorted(word_to_count.items(), key = lambda item: item[1], reverse=True)

        # TODO: retain the top 10k words
        my_vocab = []
        for word, count in top_words_to_count[:10000]:
            my_vocab.append((word, word_to_index[word]))
        
        self.vocab = dict(my_vocab)
        return


        raise NotImplementedError

    def transform(self, X: np.ndarray) -> sp.csr_matrix:
        """
        Transform the input sentences into a sparse matrix based on the
        vocabulary obtained after fitting the vectorizer
        ! Do NOT return a dense matrix, as it will be too large to fit in memory
        :param X: np.ndarray
            Input sentences (can be either train, val or test)
        :return: sp.csr_matrix
            The sparse matrix representation of the input sentences
        """
        print("AKASH => ", X.shape)
        assert self.vocab is not None, "Vectorizer not fitted yet"
        # TODO: convert the input sentences into vectors
        sentences = X
        tokenized_sentences = [sentence.split() for sentence in sentences] 
        word_indices = []  
        for sentence in tokenized_sentences:  
            indices = [self.vocab[word] for word in sentence if word in self.vocab]  
            word_indices.append(indices)

        word_counts = []  
        for sentence in tokenized_sentences:  
            counts = defaultdict(int)  
            for word in sentence:
                if word in self.vocab:  
                    counts[self.vocab[word]] += 1  
            word_counts.append(counts)

        rows = []  
        columns = []  
        values = []  
        for i, sentence in enumerate(word_counts):  
            for word_index, count in sentence.items():  
                if word_index < len(self.vocab):  
                    rows.append(i)  
                    columns.append(word_index)  
                    values.append(count)   
                else:  
                    # Optionally log the out-of-vocabulary word  
                    # print(f"Skipping out-of-vocabulary word: {word}")
                    pass
        
        sparse_matrix = sp.csr_matrix((values, (rows, columns)), shape=(len(sentences), len(self.vocab)))
        return sparse_matrix
    
        raise NotImplementedError


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)


def get_data(
        path: str,
        seed: int
) -> Tuple[np.ndarray, np.ndarray, np.array, np.ndarray]:
    """
    Load twitter sentiment data from csv file and split into train, val and
    test set. Relabel the targets to -1 (for negative) and +1 (for positive).

    :param path: str
        The path to the csv file
    :param seed: int
        The random state for reproducibility
    :return:
        Tuple of numpy arrays - (data, labels) x (train, val) respectively
    """
    # load data
    df = pd.read_csv(path, encoding='utf-8')

    # shuffle data
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    # split into train, val and test set
    train_size = int(0.8 * len(df))  # ~1M for training, remaining ~250k for val
    train_df = df.iloc[:train_size]
    val_df = df.iloc[train_size:]
    x_train, y_train =\
        train_df['stemmed_content'].values, train_df['target'].values
    x_val, y_val = val_df['stemmed_content'].values, val_df['target'].values
    return x_train, y_train, x_val, y_val
