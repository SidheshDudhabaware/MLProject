import sys
from dataclasses import dataclass
import os

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

from src.logger import logging

from src.utils import save_object

@dataclass 
class DataTransformationConfig:
    preprocesssor_ob_file_path=os.path.join('artifacts',"preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()
    
    def get_data_transformer_obj(self):
        try:
            numerical_columns=["writing_score","reading_score"]
            categorical_column=[
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course"
            ]

            num_pipeline=Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy='median')),
                    ("scaler",StandardScaler())
                ]
            )

            cat_pipeline=Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy='most_frequent')),
                    ("one_hot_encoder",OneHotEncoder()),
                    ("scaler",StandardScaler(with_mean=False))
                ]
            )

            logging.info("numerical colums standard scaler completed")
            logging.info("categorical colums encoding completed")

            preprocessor=ColumnTransformer(
                [
                    ("num_pipeline",num_pipeline,numerical_columns),
                    ("cat_pipeline",cat_pipeline,categorical_column)
                ]
            )

            return preprocessor

        except Exception as e:
            print(e)
    
    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("Read train and test data completed")
            logging.info("Obtaining preprocessing object")

            preprocessing_obj=self.get_data_transformer_obj()

            target_column_name="math_score"
            numerical_colums=["writing_score","reading_score"]

            input_feature_train_df=train_df.drop(columns=[target_column_name])
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=[target_column_name])
            target_feature_test_df=test_df[target_column_name]

            logging.info("Applying the preproccisng training and testing")

            input_features_trained_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_features_test_arr=preprocessing_obj.transform(input_feature_test_df)

            train_arr=np.c_[
                input_features_trained_arr,np.array(target_feature_train_df)
            ]

            test_arr=np.c_[
                input_features_test_arr,np.array(target_feature_test_df)
            ]

            logging.info("Saving preprocessed objects")

            save_object(
                file_path=self.data_transformation_config.preprocesssor_ob_file_path,
                obj=preprocessing_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocesssor_ob_file_path
            )

        except Exception as e:
            print(e)