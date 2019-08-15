# Matlab2Python

BIMEF accepts a numpy array of uint8 integers as the image input and outputs a numpy array of uint8 integers.

Although we convert images to RGB before applying BIMEF, BIMEF should apply the same brightening regardless of format.

BIMEF will solve an equation of the form A*x=b for a large, sparse matrix A.

We use the sksparse module to solve this equation.

How to install sksparse module for MacOSX (assuming you have numpy and scipy installed):
1. Install cholmod (run the command 'brew install suite-sparse' in the terminal).
2. Install scikit-sparse (run the command 'pip install scikit-sparse' in the terminal).

How to install sksparse module for Linux:
1. Install cholmod (run the command 'sudo apt-get install libsuitesparse-dev' in terminal).
2. Install scikit-sparse (run 'pip install scikit-sparse' in terminal).

Alternatively, you can install the slightly slower pyamg module (run the command 'pip install pyamg' in the terminal).
To use pyamg, you can comment out the sksparse.cholmod code and uncomment the pyamg code.

# Python vs MATLAB discrepancies

1. Python's cv2.resize and MATLAB's imresize resize images with slightly different results. We originally used 
an exact Python translation of MATLAB, but using cv2.resize runs ~25% faster with insignificant visual changes
to the enhanced output.

2. Python's cv2.imread and MATLAB's imread read slightly differently. Corresponding uint8 pixel values can differ 
by up to 5 or in rare cases, even over 5.

3. Python's fminbound and MATLAB's fminbnd find slightly different values of k that minimize entropy, 
probably because the Python version converges differently.

Fortunately, the enhanced images produced by both algorithms are visually indistinguishable.
