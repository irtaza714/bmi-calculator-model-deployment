import sys
import os
from dataclasses import dataclass
import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler, OneHotEncoder
from sklearn.decomposition import PCA
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
import pickle

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts',"preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformer_object(self):

        try:
            outliers = ['Height_In_Inches', 'Weight_In_Pounds']
            
            no_outliers =  ['Age']

            cat = ['Gender']
            
            outliers_pipeline= Pipeline( steps=
                                        [("imputer",SimpleImputer(missing_values = np.nan, strategy="median")),
                                         ("rs", RobustScaler())] )
            
            no_outliers_pipeline = Pipeline( steps=
                                        [
                                         ("ss", StandardScaler())] )

            cat_pipeline = Pipeline( steps=
                                  [ ('ohe', OneHotEncoder())
                                   ])
            
            preprocessor = ColumnTransformer(
                [
                    ("outliers_pipeline", outliers_pipeline, outliers),
                    ("no_outliers_pipeline", no_outliers_pipeline, no_outliers),
                    ("cat_pipeline", cat_pipeline, cat)
                ]
            )


            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,train_path,test_path):

        try:
            train = pd.read_csv(train_path)

            logging.info("Read train data")
            
            test = pd.read_csv(test_path)

            logging.info("Read test data")

            x_train_transf = train.drop('BMI',axis=1)

            logging.info("Dropped target column from the train set to make the input data frame for model training")

            y_train_transf = train['BMI']

            logging.info("Target feature obtained for model training")

            constant_columns_x_train_transf = x_train_transf.columns[x_train_transf.nunique() == 1]

            if len(constant_columns_x_train_transf) > 0:
                 print("Constant columns found in x_train_transf:", constant_columns_x_train_transf)
            else:
                print("No constant columns found in x_train_transf.")

            logging.info("Checked for constant columns in x_train_transf") 

            x_test_transf = test.drop('BMI', axis=1)

            logging.info("Dropped target column from the test set to make the input data frame for model testing")

            constant_columns_x_test_transf = x_test_transf.columns[x_test_transf.nunique() == 1]

            if len(constant_columns_x_test_transf) > 0:
                 print("Constant columns found in x_test_transf:", constant_columns_x_test_transf)
            else:
                print("No constant columns found in x_test_transf.")

            logging.info("Checked for constant columns in x_test_transf")      
        
            y_test_transf = test['BMI']

            logging.info("Target feature obtained for model testing")

            preprocessor = self.get_data_transformer_object()
            
            logging.info("Preprocessing object obtained")

            x_train_transf_preprocessed = preprocessor.fit_transform(x_train_transf)

            logging.info("Preprocessor applied on x_train_transf")

            x_train_transf_preprocessed_df = pd.DataFrame(x_train_transf_preprocessed)

            logging.info("x_train_transf dataframe formed")

            for i in range(len(x_train_transf_preprocessed_df.columns)):
                
                x_train_transf_preprocessed_df = x_train_transf_preprocessed_df.rename(columns={x_train_transf_preprocessed_df.columns[i]: f'c{i+1}'})

            logging.info("x_train_transf dataframe columns renamed")

            print ("x_train_transf_preprocessed_df shape before be:", x_train_transf_preprocessed_df.shape)

            # print ("x_train_transf_preprocessed_df columns before be:", x_train_transf_preprocessed_df.columns)

            constant_columns_x_train_transf_preprocessed_df = x_train_transf_preprocessed_df.columns[x_train_transf_preprocessed_df.nunique() == 1]

            if len(constant_columns_x_train_transf_preprocessed_df) > 0:
                 print("Constant columns found in x_train_transf_preprocessed_df:", constant_columns_x_train_transf_preprocessed_df)
            else:
                print("No constant columns found in x_train_transf_preprocessed_df.")

            x_test_transf_preprocessed = preprocessor.transform(x_test_transf)

            logging.info("Preprocessor applied on x_test_transf")

            x_test_transf_preprocessed_df = pd.DataFrame(x_test_transf_preprocessed)

            logging.info("x_test_transf dataframe formed")

            for i in range(len(x_test_transf_preprocessed_df.columns)):
                
                x_test_transf_preprocessed_df = x_test_transf_preprocessed_df.rename(columns={x_test_transf_preprocessed_df.columns[i]: f'c{i+1}'})

            logging.info("x_test_transf dataframe columns renamed")

            constant_columns_x_test_transf_preprocessed_df = x_test_transf_preprocessed_df.columns[x_test_transf_preprocessed_df.nunique() == 1]

            if len(constant_columns_x_test_transf_preprocessed_df) > 0:
                 print("Constant columns found in x_test_transf_preprocessed_df:", constant_columns_x_test_transf_preprocessed_df)
            else:
                print("No constant columns found in x_test_transf_preprocessed_df.")

            train_arr = np.c_[np.array(x_train_transf_preprocessed_df), np.array(y_train_transf)]
            
            logging.info("Combined the input features and target feature of the train set as an array.")
            
            test_arr = np.c_[np.array(x_test_transf_preprocessed_df), np.array(y_test_transf)]
            
            logging.info("Combined the input features and target feature of the test set as an array.")
            
            save_object(
            file_path=self.data_transformation_config.preprocessor_obj_file_path,
            obj=preprocessor)
            
            logging.info("Saved preprocessing object.")
            
            return (
            train_arr,
            test_arr,
            self.data_transformation_config.preprocessor_obj_file_path,)
        
        except Exception as e:
            raise CustomException(e, sys)