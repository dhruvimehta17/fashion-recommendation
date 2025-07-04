from itertools import product
import os
import requests
import pandas as pd
import ast
import numpy as np
from flask import Flask, render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.preprocessing import image
from io import BytesIO
import pickle
import re
# from cassandra_utils import load_data_from_cassandra

app = Flask(__name__)

categories = ['ACCESSORIES', 'BAGS_BACKPACKS', 'BEAUTY', 'BLAZERS', 'HOODIES_SWEATSHIRTS', 'JEANS', 'LINEN', 
              'OVERSHIRTS', 'PERFUMES', 'POLO_SHIRTS', 'SHIRTS', 'SHOES', 'SHORTS', 'SUITS', 'SWEATERS_CARDIGANS', 
              'SWIMWEAR', 'T-SHIRTS', 'TROUSERS', 'ZARA_ATHLETICZ', 'ZARA_ORIGINS']

formal_keywords = {'formal', 'office', 'meeting', 'work'}
activity_keywords = {
    'party': ['T-SHIRTS', 'JEANS', 'BLAZERS', 'SHOES', 'PERFUMES'],
    'date': ['T-SHIRTS', 'JEANS', 'BLAZERS', 'SHOES', 'PERFUMES'],
    'run': ['ZARA_ATHLETICZ', 'SHOES'],
    'gym': ['ZARA_ATHLETICZ', 'SHOES'],
    'swim': ['SWIMWEAR'],
}


def get_extra_categories(query):
    query = query.lower()
    extra = set()

    if any(word in query for word in ['formal', 'interview', 'office', 'business', 'meeting']):
        extra.update(['BLAZERS', 'SHIRTS', 'TROUSERS', 'SHOES', 'SUITS'])

    if 'party' in query or 'date' in query:
        extra.add('PERFUMES')

    if any(word in query for word in ['run', 'jog', 'gym', 'exercise', 'workout']):
        extra.add('ZARA_ATHLETICZ')

    if 'swim' in query or 'beach' in query:
        extra.add('SWIMWEAR')

    return list(extra)


def load_category_data(category):
    file_path = f"zara-dataset/men/{category}.csv"
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        df.columns = df.columns.str.strip()
        return df
    return None


def filter_products_with_images(df):
    def get_first_image(img_str):
        if isinstance(img_str, str):
            try:
                imgs = ast.literal_eval(img_str)
                if imgs and isinstance(imgs, list):
                    first_img_dict = imgs[0]
                    if isinstance(first_img_dict, dict):
                        return list(first_img_dict.keys())[0]
            except Exception:
                return None
        return None

    df['product_images'] = df['product_images'].apply(get_first_image)
    df = df[df['product_images'].notna() & (df['product_images'] != '')]
    return df


resnet_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')


def extract_image_features(img_file):
    img = image.load_img(img_file, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    features = resnet_model.predict(img_array)
    return features


def calculate_text_similarity(query_text, products):
    vectorizer = TfidfVectorizer()
    descriptions = [product['product_name'] + ' ' + product['details'] for product in products]
    tfidf_matrix = vectorizer.fit_transform(descriptions)
    query_vector = vectorizer.transform([query_text])
    similarities = cosine_similarity(query_vector, tfidf_matrix)
    return similarities[0]


def extract_keywords(text):
    tokens = re.findall(r'\b\w+\b', text.lower())
    keywords = [t for t in tokens if len(t) > 2]
    return keywords


def product_matches_keywords(product, keywords):
    text = (product['product_name'] + ' ' + product['details']).lower()
    return any(k in text for k in keywords)


def download_image(url):
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        return BytesIO(response.content)
    except requests.exceptions.RequestException as e:
        print(f"Error downloading image: {e}")
        return None


def normalize_vector(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


@app.route('/')
def index():
    category = 'SHIRTS'
    df = load_category_data(category)
    if df is None:
        return "Category data not found!"
    df = filter_products_with_images(df)
    products = df[['product_name', 'link', 'product_images', 'price', 'details']].to_dict(orient='records')
    return render_template('index.html', products=products, category=category, categories=categories, similarity_results=False, show_similarity=False)


@app.route('/category/<category>')
def category_view(category):
    df = load_category_data(category)
    if df is None:
        return "Category data not found!"
    df = filter_products_with_images(df)
    products = df[['product_name', 'link', 'product_images', 'price', 'details']].to_dict(orient='records')
    return render_template('index.html', products=products, category=category, categories=categories, similarity_results=False, show_similarity=False)


# @app.route('/search', methods=['POST'])
# def search():
#     query_text = request.form.get('query', '').strip().lower()
#     if not query_text:
#         return "Please enter a valid query."

#     query_keywords = extract_keywords(query_text)
#     all_products = load_data_from_cassandra()

#     product_dicts = []
#     for row in all_products:
#         product_dicts.append({
#             'product_name': row.product_name,
#             'details': row.details if hasattr(row, 'details') else '',
#             'price': row.price if hasattr(row, 'price') else '',
#             'link': row.link if hasattr(row, 'link') else '',
#             'product_images': row.image_url if hasattr(row, 'image_url') else '',
#             'category': row.category if hasattr(row, 'category') else ''
#         })

#     filtered_products = [p for p in product_dicts if product_matches_keywords(p, query_keywords)]
#     if not filtered_products:
#         filtered_products = product_dicts

#     try:
#         similarities = calculate_text_similarity(query_text, filtered_products)
#         for i, sim in enumerate(similarities):
#             filtered_products[i]['similarity'] = sim
#         filtered_products.sort(key=lambda x: x['similarity'], reverse=True)
#     except Exception as e:
#         print(f"TF-IDF error: {e}")
#         return "Failed to calculate similarity."

#     return render_template('index.html',
#                            products=filtered_products,
#                            category=f"Search: {query_text}",
#                            categories=categories,
#                            similarity_results=True,
#                            show_similarity=True)


@app.route('/by_text', methods=['POST'])
def by_text():
    query_text = request.form.get('query_text')
    category = request.form.get('category')
    if not category or category not in categories:
        category = 'SHIRTS'
    df = load_category_data(category)
    if df is None:
        return "Category data not found!"
    df = filter_products_with_images(df)
    products = df[['product_name', 'link', 'product_images', 'price', 'details']].to_dict(orient='records')

    keywords = extract_keywords(query_text)
    filtered_products = [p for p in products if product_matches_keywords(p, keywords)]

    if not filtered_products:
        filtered_products = products

    similarities = calculate_text_similarity(query_text, filtered_products)
    results = []
    for product, sim in zip(filtered_products, similarities):
        product_copy = dict(product)
        product_copy['similarity'] = sim
        results.append(product_copy)
    results.sort(key=lambda x: x['similarity'], reverse=True)
    return render_template('index.html', products=results, category=category, categories=categories, similarity_results=True, show_similarity=False)


@app.route('/by_image', methods=['POST'])
def by_image():
    query_image_url = request.form['query_image_url']
    category = request.form['category']

    query_image_file = download_image(query_image_url)
    if query_image_file is None:
        return "Failed to download image."

    query_features = extract_image_features(query_image_file)
    query_features = query_features.flatten()

    try:
        with open(f'features_{category}.pkl', 'rb') as f:
            cached_features = pickle.load(f)
    except FileNotFoundError:
        return "No cached image features found. Please precompute features first."

    df = load_category_data(category)
    if df is None:
        return "Category data not found."
    df = filter_products_with_images(df)
    products = df[['product_name', 'link', 'product_images', 'price', 'details']].to_dict(orient='records')

    similar_products = []
    threshold = 0.85
    for i, feat in enumerate(cached_features):
        feat_flat = feat.flatten()
        sim = np.dot(query_features, feat_flat) / (np.linalg.norm(query_features) * np.linalg.norm(feat_flat))
        sim = float(sim)
        if sim >= threshold:
            prod = products[i].copy()
            prod['similarity'] = round(sim, 2)
            similar_products.append(prod)

    return render_template(
        'index.html',
        products=similar_products,
        query_image_url=query_image_url,
        category=category,
        categories=categories,
        similarity_results=True
    )


if __name__ == '__main__':
    app.run(debug=True)
