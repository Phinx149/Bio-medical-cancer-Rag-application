# download_nltk.py
import nltk
import os

# Define the local path where NLTK data will be stored
local_nltk_data_path = os.path.join(os.getcwd(), 'nltk_data')
os.makedirs(local_nltk_data_path, exist_ok=True)

# Set NLTK_DATA environment variable for this session, and add to NLTK's path
os.environ['NLTK_DATA'] = local_nltk_data_path
if local_nltk_data_path not in nltk.data.path:
    nltk.data.path.insert(0, local_nltk_data_path)

print(f"Downloading NLTK data to: {local_nltk_data_path}")

# List of NLTK packages we've identified as necessary
nltk_packages = [
    'punkt',
    'averaged_perceptron_tagger',
    'punkt_tab',
    'averaged_perceptron_tagger_english'
]

for package in nltk_packages:
    print(f"Attempting to download '{package}'...")
    try:
        nltk.download(package, download_dir=local_nltk_data_path)
        print(f"Successfully downloaded '{package}'.")
    except Exception as e:
        print(f"ERROR: Could not download '{package}'. Please check your internet connection or NLTK version. Error: {e}")

print("\nVerifying downloads locally...")
try:
    nltk.data.find('tokenizers/punkt')
    print("Verification: 'punkt' found.")
    nltk.data.find('taggers/averaged_perceptron_tagger')
    print("Verification: 'averaged_perceptron_tagger' found.")
    nltk.data.find('tokenizers/punkt_tab')
    print("Verification: 'punkt_tab' found.")
    nltk.data.find('taggers/averaged_perceptron_tagger_english')
    print("Verification: 'averaged_perceptron_tagger_english' found.")
except LookupError as e:
    print(f"Verification FAILED for some packages after download: {e}")
    print("Ensure the 'nltk_data' folder has the correct subdirectories and files.")

print("\nLocal NLTK data setup complete. You should now commit the 'nltk_data' folder to your Git repository.")