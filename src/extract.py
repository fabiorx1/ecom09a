import zipfile
import os

def extract_data():
    zip_fpath = 'ConsultasDataset.zip'
    target_path = 'data'
    if not os.path.exists(target_path):
        os.makedirs(target_path)

    # Extract the .zip file
    with zipfile.ZipFile(zip_fpath, 'r') as zip_ref:
        zip_ref.extractall(target_path)

if __name__ == '__main__':
    extract_data()