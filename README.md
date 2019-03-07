This is a machine learning library developed by Janaan Lake for CS5350 at the University of Utah.

The LMS.py library contains Least Means Square Regression algorithms using Batch Gradient Descent and Stochastic Gradient Descent.  Currently, they are used on the concrete slump dataset found at https://archive.ics.uci.edu/ml/datasets/Concrete+Slump+Test.

To run these algorithms, run the following script on the command line: 
python3 LMS.py

Alternatively, run the run.sh shell script.  

This will run both stochastic and gradient descent on the training and test data until convergence.  The results will be shown in two graphs plotting the loss over the number of iterations.
