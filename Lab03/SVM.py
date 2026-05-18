import numpy as np 
from sklearn.metrics import precision_score, recall_score, f1_score
from tqdm import tqdm 

class SVM: 
    def __init__(self, C = 1.0, lr = 0.01, n_iterations = 1000): 
        self.C = C 
        self.lr = lr 
        self.n_iterations = n_iterations 
        self.W = None 
        self.b = None 

    def fit(self, X, y): 
        N, dim = X.shape 
        self.W = np.zeros(dim) 
        self.b = 0 
        
        pbar = tqdm(range(self.n_iterations), desc = "Training Custom SVM") 
        for epoch in pbar: 
            for i in range(N): 
                x_i = X[i] 
                y_i = y[i] 
                
                y_pred = np.dot(x_i, self.W) + self.b 
                
                if y_i * y_pred >= 1: 
                    dW = self.W / N 
                    db = 0 
                else: 
                    dW = (self.W / N) + self.C * (-y_i * x_i) 
                    db = self.C * (-y_i) 
                
                self.W -= self.lr * dW 
                self.b -= self.lr * db 

    def predict(self, X): 
        return np.dot(X, self.W) + self.b 

    def get_metrics(self, X, y): 
        y_pred_raw = self.predict(X) 
        y_pred = np.where(y_pred_raw >= 0, 1, -1) 
        
        p = precision_score(y, y_pred, zero_division=0) 
        r = recall_score(y, y_pred, zero_division=0) 
        f1 = f1_score(y, y_pred, zero_division=0) 
        
        return {"Precision": p, "Recall": r, "F1": f1}
