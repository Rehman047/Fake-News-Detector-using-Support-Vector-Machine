# Fake News Detector using Support Vector Machine

## Project Description
This project implements a Fake News Detector using a Support Vector Machine (SVM) model and a custom TF-IDF vectorizer. The goal is to classify news articles as either real or fake based on their content. This utilizes advanced machine learning techniques to identify patterns and features that differentiate authentic news articles from misleading ones.

## Features
- **Machine Learning Model**: Implements a robust SVM model with optimized parameters.
- **Data Preprocessing**: Cleans and preprocesses the text data for effective model training.
- **Performance Metrics**: Provides an accuracy report and other performance metrics post-evaluation.


## Usage Examples
You can input the article text into the user interface and click the 'Classify' button to determine if it is real or fake. Here are some examples:

- **Example 1**: "The government has issued a new policy to combat climate change."
- **Example 2**: "Aliens have landed on Earth and are taking over the planet!"

Upon inputting these articles, the model will classify the first as 'Real' and the second as 'Fake'.

## Technical Details
- **Technology Stack**: Python, Scikit-learn, Pandas, Numpy, MatplotLib
- **Data Sources**: Utilizes data from publicly available datasets to train the model.
- **Model Training**: The SVM model is trained on a labeled dataset with various features extracted from the text.
- **Evaluation**: The model is evaluated using metrics such as precision, recall, and F1 score to ensure reliability.



## How It Works
1. **Data Collection**: News articles are collected and labeled as real or fake.
2. **Feature Extraction**: Text features are extracted using custom TF-IDF vectorization.
3. **Model Training**: The SVM classifier is trained on the extracted features.
4. **Classification**: New articles are classified based on the trained model.
5. **Results**: The model returns a prediction indicating whether the article is real or fake.



## Future Enhancements
- Implement deep learning models for improved accuracy.
- Add support for multiple languages.
- Develop a web-based interface for easier accessibility.
- Integrate with news APIs for real-time detection.

