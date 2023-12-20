# -*- coding: utf-8 -*-
"""
Created on Thu Jun 17 15:27:20 2021

@author: Kaike Sa Teles Rocha Alves
@email: kaike.alves@engenharia.ufjf.br
"""
# Importing libraries
import pandas as pd
import numpy as np

class ADA_KRLS:
    def __init__(self, M = 100, nu = 0.1, sigma = 0.1, tau = 0.01):
        # H is equivalent to P in KRLS
        self.parameters = pd.DataFrame(columns = ['Kinv', 'alpha', 'H', 'm', 'Dict', 'yn'])
        # Computing the output in the training phase
        self.OutputTrainingPhase = np.array([])
        # Computing the residual square in the ttraining phase
        self.ResidualTrainingPhase = np.array([])
        # Computing the output in the testing phase
        self.OutputTestPhase = np.array([])
        # Computing the residual square in the testing phase
        self.ResidualTestPhase = np.array([])
        # Hyperparameters and parameters
        self.sigma = sigma
        # Threshold of ALD sparse rule
        self.nu = nu
        # Maximum number of inputs in the dictionary
        self.M = M
        # Tau serves to avoid misalignment
        self.tau = tau
         
    def fit(self, X, y):

        # Compute the number of samples
        n = X.shape[0]
        
        # Initialize the first input-output pair
        x0 = X[0,].reshape(-1,1)
        y0 = y[0]
        
        # Initialize ADA_KRLS
        self.Initialize_ADA_KRLS(x0, y0)

        for k in range(1, n):

            # Prepare the k-th input vector
            x = X[k,].reshape((1,-1)).T
                      
            # Update ADA_KRLS
            k_til = self.ADA_KRLS(x, y[k])
            
            # Compute output
            Output = self.parameters.loc[0, 'alpha'].T @ k_til
            
            # Store results
            self.OutputTrainingPhase = np.append(self.OutputTrainingPhase, Output )
            self.ResidualTrainingPhase = np.append(self.ResidualTrainingPhase,(y[k]) - Output )
        return self.OutputTrainingPhase
            
    def predict(self, X):

        for k in range(X.shape[0]):
            
            # Prepare the first input vector
            x = X[k,].reshape((1,-1)).T

            # Compute k
            k_til = np.array(())
            for ni in range(self.parameters.loc[0, 'Dict'].shape[1]):
                k_til = np.append(k_til, [self.Kernel(self.parameters.loc[0, 'Dict'][:,ni].reshape(-1,1), x)])
            k_til = k_til.reshape(-1,1)
            
            # Compute the output
            Output = self.parameters.loc[0, 'alpha'].T @ k_til
            
            # Store the output
            self.OutputTestPhase = np.append(self.OutputTestPhase, Output )

        return self.OutputTestPhase

    def Kernel(self, x1, x2):
        k = np.exp( - ( 1/2 ) * ( (np.linalg.norm( x1 - x2 ))**2 ) / ( self.sigma**2 ) )
        return k
    
    def Initialize_ADA_KRLS(self, x, y):
        k11 = self.Kernel(x, x)
        Kinv = np.ones((1,1)) / ( k11 )
        alpha = np.ones((1,1)) * y / k11
        yn = np.eye(1) * y
        NewRow = pd.DataFrame([[Kinv, alpha, np.ones((1,1)), 1., x, yn]], columns = ['Kinv', 'alpha', 'H', 'm', 'Dict', 'yn'])
        self.parameters = pd.concat([self.parameters, NewRow], ignore_index=True)
        # Initialize first output and residual
        self.OutputTrainingPhase = np.append(self.OutputTrainingPhase, y)
        self.ResidualTrainingPhase = np.append(self.ResidualTrainingPhase, 0.)
        
    def ADA_KRLS(self, x, y):
        i = 0
        # Compute k
        k_til = np.array(())
        for ni in range(self.parameters.loc[i, 'Dict'].shape[1]):
            k_til = np.append(k_til, [self.Kernel(self.parameters.loc[i, 'Dict'][:,ni].reshape(-1,1), x)])
        k_til = k_til.reshape(-1,1)
        # Compute A
        A = self.parameters.loc[i, 'Kinv'] @ k_til
        delta = self.Kernel(x, x) - ( k_til.T @ A ).item()
        if delta == 0:
            delta = 1.
        # Estimating the error
        EstimatedError = ( y - np.matmul(k_til.T, self.parameters.loc[i, 'alpha']) ).item()
        # Novelty criterion
        if delta > self.nu and self.parameters.loc[i, 'Dict'].shape[1] < self.M:
            # Update the dictionary
            self.parameters.at[i, 'Dict'] = np.hstack([self.parameters.loc[i, 'Dict'], x])
            # Update yn
            self.parameters.at[i, 'yn'] = np.vstack([self.parameters.loc[i, 'yn'], y])
            # Update m
            self.parameters.at[i, 'm'] = self.parameters.loc[i, 'm'] + 1
            # Update Kinv                      
            self.parameters.at[i, 'Kinv'] = (1/delta)*(self.parameters.loc[i, 'Kinv'] * delta + np.matmul(A, A.T))
            self.parameters.at[i, 'Kinv'] = np.lib.pad(self.parameters.loc[i, 'Kinv'], ((0,1),(0,1)), 'constant', constant_values=(0))
            sizeKinv = self.parameters.loc[i,  'Kinv'].shape[0] - 1
            self.parameters.at[i, 'Kinv'][sizeKinv,sizeKinv] = (1/delta)
            self.parameters.at[i, 'Kinv'][0:sizeKinv,sizeKinv] = (1/delta)*(-A.flatten())
            self.parameters.at[i, 'Kinv'][sizeKinv,0:sizeKinv] = (1/delta)*(-A.flatten())
            # Update H
            self.parameters.at[i, 'H'] = np.lib.pad(self.parameters.loc[i, 'H'], ((0,1),(0,1)), 'constant', constant_values=(0))
            sizeP = self.parameters.loc[i,  'H'].shape[0] - 1
            self.parameters.at[i, 'H'][sizeP,sizeP] = 1.
            # Update alpha
            self.parameters.at[i, 'alpha'] = self.parameters.loc[i, 'alpha'] - ( ( A / delta ) * EstimatedError )
            self.parameters.at[i, 'alpha'] = np.vstack([self.parameters.loc[i, 'alpha'], ( 1 / delta ) * EstimatedError ])
            k_til = np.append(k_til, self.Kernel(x, x).reshape(1,1), axis=0)
            
        # Verify if the size of the dictionary is greater than M
        elif delta > self.nu and self.parameters.loc[i, 'Dict'].shape[1] > self.M:
            # Update dictionary
            self.parameters.at[i, 'Dict'] = np.hstack([self.parameters.loc[i, 'Dict'], x])
            # Update yn
            self.parameters.at[i, 'yn'] = np.vstack([self.parameters.loc[i, 'yn'], y])
            # Compute k
            kn1 = k_til
            knn = self.Kernel(x, x)
            k_til = np.append(k_til, knn)
            # Compute Dinv
            D_inv = self.parameters.loc[i, 'Kinv']
            # Update Kinv
            g = 1 / ( ( knn + self.lambda1 ) - kn1.T @ D_inv @ kn1 )
            f = ( - D_inv @ kn1 * g ).flatten()
            E = D_inv - D_inv @ kn1 @ f.reshape(1,-1)
            sizeKinv = self.parameters.loc[i,  'Kinv'].shape[0] - 1
            #self.parameters.at[i,  'Kinv'] = ( D_inv @ ( np.eye(sizeKinv + 1) + kn1 @ kn1.T @ D_inv * g ) ) 
            self.parameters.at[i, 'Kinv'] = E 
            self.parameters.at[i, 'Kinv'] = np.lib.pad(self.parameters.loc[i, 'Kinv'], ((0,1),(0,1)), 'constant', constant_values=(0))
            sizeKinv = self.parameters.loc[i,  'Kinv'].shape[0] - 1
            self.parameters.at[i, 'Kinv'][sizeKinv,sizeKinv] = g
            self.parameters.at[i, 'Kinv'][0:sizeKinv,sizeKinv] = f
            self.parameters.at[i, 'Kinv'][sizeKinv,0:sizeKinv] = f
            
            # Compute alpha
            self.parameters.at[i, 'alpha'] = self.parameters.loc[i,  'Kinv'] @ self.parameters.loc[i,  'yn']
            alpha = self.parameters.loc[i, 'alpha']
            # Find the diagonal of Kinv
            diag = np.diagonal(self.parameters.loc[i,  'Kinv'])
            d = np.zeros(diag.shape)
            for row in range(d.shape[0]):
                if diag[row] != 0:
                    d[row] = abs(alpha[row])/ diag[row]
                else:
                    d[row] = abs(alpha[row])
            j = d.argmin()
            # Remove the least relevant element in the dictionary
            self.parameters.at[i,  'Dict'] = np.delete(self.parameters.loc[i,  'Dict'], j, 1)
            # Update yn
            self.parameters.at[i,  'yn'] = np.delete(self.parameters.loc[i,  'yn'], j, 0)
            # Update k
            k_til = np.delete(k_til, j)
            # Number of elements in Kinv
            idx = np.arange(self.parameters.loc[i,  'Kinv'].shape[1])
            noj = np.delete(idx, j)
            # Compute Dinv
            G = self.parameters.loc[i,  'Kinv'][noj, :][:, noj]
            f = self.parameters.loc[i,  'Kinv'][noj, j].reshape(-1,1)
            ft = self.parameters.loc[i,  'Kinv'][j,noj].reshape(1,-1)
            e = self.parameters.loc[i,  'Kinv'][j,j]
            D_inv = G - ( f @ ft ) / e
            # Update Kinv
            self.parameters.at[i,  'Kinv'] = D_inv
            # Compute alpha
            self.parameters.at[i, 'alpha'] = self.parameters.loc[i,  'Kinv'] @ self.parameters.loc[i,  'yn']
                
        else:
            # Calculate r
            r = np.matmul( self.parameters.loc[i,  'H'], A) / ( 1 + np.matmul(np.matmul(A.T, self.parameters.loc[i, 'H']), A ) )
            # Update H
            self.parameters.at[i, 'H'] = self.parameters.loc[i, 'H'] - r @ A.T @ self.parameters.loc[i, 'H']
            # Updating alpha
            #self.parameters.at[i, 'alpha'] = self.parameters.loc[i, 'alpha'] + np.matmul(self.parameters.loc[i, 'Kinv'], r) * EstimatedError
            self.parameters.at[i, 'alpha'] = self.parameters.loc[i, 'alpha'] + ( np.matmul(self.parameters.loc[i, 'Kinv'], r) * EstimatedError ) / ( self.tau + np.linalg.norm(k_til)**2 )
        return k_til