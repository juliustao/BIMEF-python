# Matlab2Python

BIMEF accepts a numpy array of uint8 integers as the image input and outputs a numpy array of uint8 integers.

Although we convert images to RGB before applying BIMEF, BIMEF should apply the same brightening regardless of format.

BIMEF will solve an equation of the form A*x=b for a large, sparse matrix A.

We use the sksparse module to solve this equation.

How to install sksparse module (assuming you have numpy and scipy installed):
1. Install cholmod (run the command 'brew install suite-sparse' in the terminal).
2. Install scikit-sparse (run the command 'pip install scikit-sparse' in the terminal).

Alternatively, you can install the slightly slower pyamg module (run the command 'pip install pyamg' in the terminal).
To use pyamg, you can modify lines 172-184 by commenting out the sksparse.cholmod code and uncomment the pyamg code.