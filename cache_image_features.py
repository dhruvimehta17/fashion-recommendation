import pickle
import requests
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.preprocessing import image
import numpy as np
import pandas as pd
import ast
from io import BytesIO

# Initialize ResNet50 model (for feature extraction)
resnet_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')

# Function to download an image from URL
def download_image(url):
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()  # Check if the request was successful
        return BytesIO(response.content)
    except requests.exceptions.RequestException as e:
        print(f"Error downloading image: {e}")
        return None

# Function to extract image features using ResNet50
def extract_image_features(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    features = resnet_model.predict(img_array)
    return features

# Function to precompute and cache image features
def cache_image_features():
    categories = ['ACCESSORIES', 'BAGS_BACKPACKS', 'BEAUTY', 'BLAZERS', 'HOODIES_SWEATSHIRTS', 'JEANS', 'LINEN', 
                  'OVERSHIRTS', 'PERFUMES', 'POLO_SHIRTS', 'SHIRTS', 'SHOES', 'SHORTS', 'SUITS', 'SWEATERS_CARDIGANS', 
                  'SWIMWEAR', 'T-SHIRTS', 'TROUSERS', 'ZARA_ATHLETICZ', 'ZARA_ORIGINS']
    
    for category in categories:
        try:
            # Load category data (assumes CSV files are in `zara-dataset/men/` directory)
            file_path = f"zara-dataset/men/{category}.csv"
            df = pd.read_csv(file_path)
            df.columns = df.columns.str.strip()  # Strip leading/trailing spaces from column names
            
            # Check for 'product_images' column
            if 'product_images' not in df.columns:
                print(f"Column 'product_images' not found in category: {category}")
                continue
            
            df['product_images'] = df['product_images'].apply(lambda x: ast.literal_eval(x)[0].keys() if isinstance(x, str) and ast.literal_eval(x) else '')
            df['product_images'] = df['product_images'].apply(lambda x: list(x)[0] if x else None)
            df = df[df['product_images'].notna() & (df['product_images'] != '')]
            
            product_features = []
            for product in df[['product_name', 'product_images']].to_dict(orient='records'):
                img_path = download_image(product['product_images'])
                if img_path:
                    features = extract_image_features(img_path)
                    product_features.append(features)
            
            # Save the product features into a .pkl file for this category
            with open(f'features_{category}.pkl', 'wb') as f:
                pickle.dump(product_features, f)
            
            print(f"Features for category '{category}' saved successfully.")

        except FileNotFoundError:
            print(f"File for category '{category}' not found. Skipping.")
        except Exception as e:
            print(f"An error occurred while processing category '{category}': {e}")

# Run the function to precompute features and save them as pickle files
cache_image_features()
