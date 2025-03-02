import os                                                                                  
import pickle                                                                               
import pandas as pd                                                                                                                                                 
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE

class FraudDetection: 
    
    def __init__(self, dataset_path=None, model_filename="fraud_detection_model.pkl"):
        self.dataset_path = dataset_path
        self.model_filename = model_filename
        self.model = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42, max_depth=10, class_weight='balanced')
        self.scaler = RobustScaler()
        self.data = None
        
    def loadData(self):
        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(f"The file '{self.dataset_path}' was not found.")
        self.data = pd.read_csv(self.dataset_path)
        
    def exploreData(self):
        #explore dataset
        df = self.data
        df.info()
        df.head()
        df.describe()
        
    def preprocessData(self):
        #split features and target
        X = self.data.drop(['Class'], axis=1)
        y = self.data['Class']
        
        # Handle class imbalance using SMOTE
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y)
        
        #split into test and train 
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)
        
        # Scale features
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
    def trainModel(self):
        #load model and show progress
        print("Loading Model...")
        self.model.fit(self.X_train_scaled, self.y_train)
        print("Model training complete.")
        print("Saving Model....")
        self.saveModel()
    
    #FROM CHATGPT
    def saveModel(self):
        #Save the trained model and scaler
        with open(self.model_filename, 'wb') as file:
            pickle.dump({'model': self.model, 'scaler': self.scaler}, file)
        print(f"Model and scaler saved to {self.model_filename}")
    
    #FROM CHATGPT
    def loadModel(self):
        # Load the trained model and scaler from a file
        if not os.path.exists(self.model_filename):
            raise FileNotFoundError(f"The model file '{self.model_filename}' was not found.")
        with open(self.model_filename, 'rb') as file:
            data = pickle.load(file)
            self.model = data['model']
            self.scaler = data['scaler']  # The scaler should already be fitted
    
        print(f"Model and scaler loaded from {self.model_filename}")

    def predict_fraud(self, new_data):
        new_data = new_data[self.X_train.columns]
        new_data_scaled = self.scaler.transform(new_data)
        print("Starting predictions....")
        predictions = self.model.predict(new_data_scaled)
        print("Predictions done.")
        return predictions
    
    #FROM CHATGPT
    def save_predictions(self, new_data, predictions, all_predictions_file="predictions.csv", fraud_only_file="OnlyFraud.csv"):
        #define both predictions and fraud 
        new_data['Prediction'] = predictions
        new_data['Fraud'] = new_data['Prediction'].apply(lambda x: 'Fraud' if x == 1 else 'Not Fraud')
        # Save all predictions
        new_data.to_csv(all_predictions_file, index=False)
        print(f"All predictions saved to {all_predictions_file}")

        # Save only fraud predictions
        fraud_cases = new_data[new_data['Prediction'] == 1]
        fraud_cases.to_csv(fraud_only_file, index=False)
        print(f"Fraud cases saved to {fraud_only_file}")
    
    #FROM CHATGPT    
    def evaluate_model(self):
        #get evaluations from 
        y_pred = self.model.predict(self.X_test_scaled)
        
        print("Model Evaluation Metrics:")
        print(classification_report(self.y_test, y_pred))
         
    def PieChartPost(self, data):
        class_counts = data['Class'].value_counts()
        class_labels = ['Not Fraud', 'Fraud']
        plt.figure(figsize=(6, 6))
        plt.pie(class_counts, labels=class_labels, autopct='%1.1f%%', startangle=90, colors=['skyblue', 'orange'])
        plt.title("Fraud vs Non-Fraud Predictions (Pie Chart)")
        plt.show()

    def plotGraphs(self, predictions_file):
        print("\nPreProcessing\n")
        self.exploreData()

        
        print("\nPost Training\n")
        self.evaluate_model()
        
        print("\nPost Prediction\n")
        predictions_data = pd.read_csv(predictions_file)
        self.PieChartPost(predictions_data)
        

        
if __name__ == "__main__":
    # # Initialize the fraud detection system
    fraud_detection = FraudDetection(dataset_path='creditcard.csv')
    
    # Load the dataset
    fraud_detection.loadData()

    # Preprocess the data
    fraud_detection.preprocessData()
    
    # Train the model
    fraud_detection.trainModel()
    
    #load Model
    fraud_detection.loadModel()

    # Predict fraud on new data
    prediction_df = 'datasets/dataset_100k.csv'
    new_data = pd.read_csv(prediction_df)
    predictions = fraud_detection.predict_fraud(new_data)

    # Save predictions to a file
    fraud_detection.save_predictions(new_data, predictions)
    
    # load predictions
    predictions_file = "predictions.csv"
    
    # Visualize data and model insights
    fraud_detection.plotGraphs(predictions_file)