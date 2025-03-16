import numpy as np
from scipy import sparse as sp


class MultinomialNaiveBayes:
    """
    A Multinomial Naive Bayes model
    """
    def __init__(self, alpha=0.01) -> None:
        """
        Initialize the model
        :param alpha: float
            The Laplace smoothing factor (used to handle 0 probs)
            Hint: add this factor to the numerator and denominator
        """
        self.alpha = alpha # Smoothing parameter
        self.priors = None # Prior probabilities of classes
        self.means = None
        self.i = 0  # to keep track of the number of examples seen
        self.class_conditional_means = None  # Probability of features given each class  
        self.classes = None  # Unique classes in the dataset  

    def fit(self, X: sp.csr_matrix, y: np.ndarray, update=False) -> None:
        """
        Fit the model on the training data
        :param X: sp.csr_matrix
            The training data
        :param y: np.ndarray
            The training labels
        :param update: bool
            Whether to the model is being updated with new data
            or trained from scratch
        :return: None
        """

        if update:
            """Update the model with newly labeled data."""  
            if self.priors is None:  
                self.fit(X, y)  
                return  

            n_features = X.shape[1]  

            for i, c in enumerate(self.classes):  
                X_c_new = X[y == c]  # New samples for class c  
                if X_c_new.shape[0] == 0:  
                    continue  

                # Update priorsX
                self.priors[i] = (self.priors[i] * self.i + X_c_new.shape[0]) / (self.i + X.shape[0])  

                # Update class conditional means  
                old_sum = self.class_conditional_means[i, :] * (self.i * self.priors[i] + n_features * self.alpha)  
                new_sum = X_c_new.sum(axis=0) + self.alpha  
                self.class_conditional_means[i, :] = (old_sum + new_sum) / (self.i * self.priors[i] + X_c_new.sum() + n_features * self.alpha)  

            self.i += X.shape[0]
            return  

        """  
        Fit the Bernoulli Naive Bayes model to the training data.  
        """ 
        self.classes = np.unique(y)  
        n_classes = len(self.classes)  
        n_features = X.shape[1]  

        # Initialize priors and class conditional means  
        self.priors = np.zeros(n_classes, dtype=np.float64)  
        self.class_conditional_means = np.zeros((n_classes, n_features), dtype=np.float64)  

        # Compute priors and class conditional means  
        for i, c in enumerate(self.classes):  
            X_c = X[y == c]  # Samples for class c  
            self.priors[i] = X_c.shape[0] / X.shape[0]  # Prior for class c  
            self.class_conditional_means[i, :] = (X_c.sum(axis=0) + self.alpha) / (X_c.shape[0] + 2 * self.alpha)  # Smoothed mean  

        self.i = X.shape[0]    
        return 
    
        raise NotImplementedError

    def predict(self, X: sp.csr_matrix) -> np.ndarray:
        """
        Predict the labels for the input data
        :param X: sp.csr_matrix
            The input data
        :return: np.ndarray
            The predicted labels
        """
        assert self.priors.shape[0] == self.class_conditional_means.shape[0]
        preds = []
        """  
        Predict the class labels for the input data.  
        """  
        assert self.priors is not None, "Model not fitted yet."  
        # posteriors = []
        for i in range(X.shape[0]):  
            x = X[i].toarray().flatten()  # Convert sparse matrix to dense array  
            class_probs = []  
            for j, c in enumerate(self.classes):  
                # Compute log prior  
                log_prior = np.log(self.priors[j])  

                # Compute log likelihood using Bernoulli distribution  
                log_likelihood = np.sum(x * np.log(self.class_conditional_means[j, :]) +  
                                       (1 - x) * np.log(1 - self.class_conditional_means[j, :]))  

                # Compute posterior (log prior + log likelihood)  
                class_probs.append(log_prior + log_likelihood)  

            # Select the class with the highest posterior probability  
            pred = self.classes[np.argmax(class_probs)]  
            preds.append(pred)  

        return np.array(preds)
    
    def predict_(self, X: sp.csr_matrix) -> np.ndarray:
            """
            Predict the labels for the input data
            :param X: sp.csr_matrix
                The input data
            :return: np.ndarray
                The predicted labels
            """
            assert self.priors.shape[0] == self.class_conditional_means.shape[0]
            preds = []
            """  
            Predict the class labels for the input data.  
            """  
            assert self.priors is not None, "Model not fitted yet."  
            # posteriors = []
            for i in range(X.shape[0]):  
                x = X[i].toarray().flatten()  # Convert sparse matrix to dense array  
                class_probs = []  
                for j, c in enumerate(self.classes):  
                    # Compute log prior  
                    log_prior = np.log(self.priors[j])  

                    # Compute log likelihood using Bernoulli distribution  
                    log_likelihood = np.sum(x * np.log(self.class_conditional_means[j, :]) +  
                                        (1 - x) * np.log(1 - self.class_conditional_means[j, :]))  

                    # Compute posterior (log prior + log likelihood)  
                    class_probs.append(log_prior + log_likelihood)  

                # Select the class with the highest posterior probability  
                # pred = self.classes[np.argmax(class_probs)]  
                preds.append(np.array(class_probs))  
            return np.array(preds)
    

