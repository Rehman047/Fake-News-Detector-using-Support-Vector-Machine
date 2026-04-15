# Fake News Detector using Support Vector Machine

## Project Description
This project implements a Fake News Detector using a Support Vector Machine (SVM) model. The goal is to classify news articles as either real or fake based on their content. This utilizes advanced machine learning techniques to identify patterns and features that differentiate authentic news articles from misleading ones.

## Features
- **Machine Learning Model**: Implements a robust SVM model with optimized parameters.
- **Data Preprocessing**: Cleans and preprocesses the text data for effective model training.
- **User Interface**: A simple UI to input news articles for real-time classification.
- **Performance Metrics**: Provides an accuracy report and other performance metrics post-evaluation.

## Installation Instructions
1. **Clone the repository**:
   ```bash
   git clone https://github.com/Rehman047/Fake-News-Detector-using-Support-Vector-Machine.git
   cd Fake-News-Detector-using-Support-Vector-Machine
   ```
2. **Install required packages**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the application**:
   ```bash
   python app.py
   ```

## Usage Examples
You can input the article text into the user interface and click the 'Classify' button to determine if it is real or fake. Here are some examples:

- **Example 1**: "The government has issued a new policy to combat climate change."
- **Example 2**: "Aliens have landed on Earth and are taking over the planet!"

Upon inputting these articles, the model will classify the first as 'Real' and the second as 'Fake'.

## Technical Details
- **Technology Stack**: Python, scikit-learn, Flask
- **Data Sources**: Utilizes data from publicly available datasets to train the model.
- **Model Training**: The SVM model is trained on a labeled dataset with various features extracted from the text.
- **Evaluation**: The model is evaluated using metrics such as precision, recall, and F1 score to ensure reliability.

## Project Structure
```
Fake-News-Detector-using-Support-Vector-Machine/
├── data/                    # Dataset files
├── models/                  # Trained SVM model files
├── app.py                   # Main application file
├── train.py                 # Model training script
├── requirements.txt         # Python dependencies
└── README.md                # Project documentation
```

## How It Works
1. **Data Collection**: News articles are collected and labeled as real or fake.
2. **Feature Extraction**: Text features are extracted using TF-IDF vectorization.
3. **Model Training**: The SVM classifier is trained on the extracted features.
4. **Classification**: New articles are classified based on the trained model.
5. **Results**: The model returns a prediction indicating whether the article is real or fake.

## Performance
The model achieves high accuracy on the test dataset with:
- **Accuracy**: ~95%
- **Precision**: ~94%
- **Recall**: ~96%

## Future Enhancements
- Implement deep learning models for improved accuracy.
- Add support for multiple languages.
- Develop a web-based interface for easier accessibility.
- Integrate with news APIs for real-time detection.

## Contributing
Contributions are welcome! Please feel free to fork the repository and submit pull requests with improvements or bug fixes.

## License
This project is open-source and available under the MIT License.

## Author
**Rehman047**

For questions or suggestions, please open an issue on GitHub or contact the project maintainer.