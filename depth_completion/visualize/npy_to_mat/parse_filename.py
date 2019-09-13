import os

def parse_filename(filename):
    filename = os.path.basename(filename)
    out = '_'.join(filename.split('_')[:4]) + '.mat'
    return out
