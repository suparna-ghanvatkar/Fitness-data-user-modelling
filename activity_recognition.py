'''
This script takes a transformed file as input, and also the model file saved using joblib and produces new file with activity labels.
'''

import argparse

parser = argparse.ArgumentParse()
parser.add_argument('tx', type = str, help = 'transformed featured file input path')
parser.add_argument('model', type = str, help = 'model file path saved using joblib')
args = parser.parse_args()


