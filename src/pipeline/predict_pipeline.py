import sys
import numpy as np
import pandas as pd
from src.exception import CustomException
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            model_path = "artifacts/model.pkl"
            preprocessor_path = "artifacts/preprocessor.pkl"
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            data_scaled = preprocessor.transform(features)
            data_scaled_df = pd.DataFrame(data_scaled)
            for i in range(len(data_scaled_df.columns)):  
                data_scaled_df = data_scaled_df.rename(columns={data_scaled_df.columns[i]: f'c{i+1}'})
            
            data_scaled_df_np = np.array(data_scaled_df)
            preds = model.predict(data_scaled_df_np)
            return preds

        except Exception as e:
            raise CustomException(e, sys)

class CustomData:
    def __init__(self, Gender: str, Age: int, Height_In_Inches: float, Weight_In_Pounds: float):
        
        self.Gender = Gender
        self.Age = Age
        self.Height_In_Inches = Height_In_Inches
        self. Weight_In_Pounds =  Weight_In_Pounds
        

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "Gender": [self.Gender],
                "Age": [self.Age],
                "Height_In_Inches" : [self.Height_In_Inches], 
                "Weight_In_Pounds" : [self.Weight_In_Pounds]
               }
            return pd.DataFrame(custom_data_input_dict)
        except Exception as e:
            raise CustomException(e, sys)