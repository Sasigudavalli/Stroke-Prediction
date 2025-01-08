**Stroke Risk Prediction Model**
**Overview**
This project aims to develop a machine learning-based Stroke Risk Prediction Model that can predict the likelihood of a stroke based on a patient's medical details. The model uses Logistic Regression and Support Vector Machine (SVM) to classify stroke risk. The class imbalance in the dataset is addressed using SMOTE (Synthetic Minority Over-sampling Technique), and the final solution is deployed using Streamlit for real-time prediction.

**Project Structure**
Data Preprocessing:

The dataset is loaded and cleaned. Missing values in the BMI column are filled with the mean.
Categorical variables like gender, ever_married, work_type, and Residence_type are encoded to numerical values.
The smoking_status feature is one-hot encoded.
The dataset is split into features (X) and target (Y).
Handling Class Imbalance:

The dataset is imbalanced, with a significantly lower number of stroke cases. To handle this, the SMOTETomek technique is applied to balance the dataset.
Model Building:

Logistic Regression is trained on the processed data.
The model is evaluated using accuracy and other performance metrics.
Model Deployment:

The trained model and scaler are saved using Pickle for later use.
A real-time prediction application is created using Streamlit, allowing users to input patient data and receive stroke risk predictions.
Libraries Used
Polars for efficient data manipulation.
Pandas and NumPy for data processing and manipulation.
Seaborn for data visualization.
Scikit-learn for machine learning models, preprocessing, and evaluation.
SMOTE from imbalanced-learn to handle class imbalance.
Streamlit for creating a real-time web interface.
Setup
1. Install Required Libraries
To run this project, you need the following Python libraries:

bash
Copy code
pip install polars pandas numpy seaborn scikit-learn imbalanced-learn streamlit pickle
2. Dataset
Download the dataset from Healthcare Stroke Prediction Dataset.

Ensure that the dataset is placed at:

arduino
Copy code
C:\Users\user\OneDrive\Desktop\Datasets\Stroke\healthcare-dataset-stroke-data.csv
3. Model Training
The dataset is preprocessed to clean and encode the features.
Logistic Regression and SVM models are trained.
SMOTE is applied to balance the dataset and address the class imbalance.
The model and scaler are saved using Pickle.
python
Copy code
import pickle
model_directory = r'C:\Users\user\OneDrive\Desktop\mdlpkls'
pickle.dump(model1, open(f'{model_directory}\\stroke_model.sav', 'wb'))
pickle.dump(sc1, open(f'{model_directory}\\standard_model.pkl', 'wb'))
4. Streamlit App
To run the Streamlit app, navigate to the project directory and use the following command:

bash
Copy code
streamlit run app.py
5. Web Interface
The user can input various patient details (e.g., age, gender, hypertension, etc.) in the sidebar.
The app will display the stroke risk prediction and the probability based on the input data.
Key Features
Real-Time Prediction: Users can input data and get immediate stroke risk predictions.
Data Visualization: Count plots and statistical summaries are provided to visualize class distribution and other insights.
Model Deployment: The trained model is deployed using Streamlit, providing an interactive interface for making predictions.
Future Improvements
Implementing more advanced models (e.g., Random Forest, XGBoost) for improved performance.
Adding more features or external data for enhanced prediction accuracy.
Incorporating user authentication for secure app access.
