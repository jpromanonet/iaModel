# Let's import some libraries
from gravityai import gravityai as grav
import pickle
import pandas as pd

# Let's create some pickle files
model = pickle.load(open(''))
tfidf_vectorizer = pickle.load(open(''))
label_encoder = pickle.load(open(''))

# Let's write our functions
def process(inPath, outPath):
    # read input file from CSV
    input_df = pd.read_csv(inPath)
    # Vectorize data
    features = tfidf_vectorizer.transform(input_df['body'])