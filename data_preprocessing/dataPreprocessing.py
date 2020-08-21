import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from app_logger import logger

class Preprocessor:
    """
        This class shall  be used to clean and transform the data before Prediction or Training.
    """

    def __init__(self,filepath,process_type):
        self.file_object = open("logs/PreProcessing_Log.txt", 'a+')
        self.logger_object = logger.App_Logger()

    def data_preprocess(self,filepath,process_type):

        self.logger_object.log(self.file_object, 'Pre-Processing - Started the pre-processing of the Validated file')

        try:
            if process_type == "T":
                # Reading the inputs given by the user
                df = pd.read_csv(filepath)
                self.logger_object.log(self.file_object, 'Pre-Processing - Successfully read the Re-Training file')

                # Reading the old train data
                df_old_traindata = pd.read_csv("Old_TrainData/Project_Data.csv")
                self.logger_object.log(self.file_object, 'Re-Training - Successfully read the Old Train Data.')

                # Merging the Old Train Data with the inputs given by the user
                df_retrain = pd.concat([df, df_old_traindata], axis=0)

                # Converting Multi class problem into binary class since 1 is having disease and rest all [2 to 5] doesn't has disease
                y_val = {1: 1, 2: 0, 3: 0, 4: 0, 5: 0}
                df_retrain['target_var'] = df_retrain['y'].map(y_val)
                df_retrain.drop(['y'], axis=1, inplace=True)

                # Splitting the dataset into X & Y
                df_X = df_retrain.drop(['target_var'], axis=1)
                df_Y = df_retrain.target_var
                df_Y.to_csv('Training_Files/Good_Raw/df_Y.csv', index=False)
                self.logger_object.log(self.file_object, 'Pre-Processing - Successfully saved the df_Y file desired location.')

            if process_type == "P":
                #  reading the inputs given by the user
                df_X = pd.read_csv(filepath)
                self.logger_object.log(self.file_object, 'Pre-Processing - Successfully read the Prediction file.')


            # Saving the Patient ID into another file.
            df_X.rename(columns={df_X.columns[0]:"Patient_ID"},inplace=True)
            df_patient_id = df_X.Patient_ID
            df_patient_id.to_csv('Preprocessed_Files/df_patient_id.csv', index=False)
            self.logger_object.log(self.file_object, 'Pre-Processing - Successfully saved the patient ID file at desired location')

            # Dropping the '1st column - "unnamed:0"
            df_X.drop(df_X.columns[0], axis=1, inplace=True)
            self.logger_object.log(self.file_object, 'Pre-Processing - Successfully removed the 1st Column''.')

            #  Scale the dataset using Satandard scaler
            scalar = StandardScaler()
            df_X_scaled = scalar.fit_transform(df_X)
            self.logger_object.log(self.file_object, 'Pre-Processing - Successfully scaled down the data.')

            df_X_scaled = pd.DataFrame(df_X_scaled, columns=df_X.columns)

            # PCA on standardized data
            if process_type == "T":
                pca = PCA()
                X_stand_pca = pca.fit_transform(df_X_scaled)
                df_X_stand_pca = pd.DataFrame(X_stand_pca)
                self.logger_object.log(self.file_object,
                                           'Pre-Processing - Successfully transformed the data using PCA.')
            if process_type == "T":
                df_X_stand_pca.to_csv('Preprocessed_Files/Preprocessed_File.csv', index=False)
            if process_type == "P":
                df_X_scaled.to_csv('Preprocessed_Files/Preprocessed_File.csv', index=False)

            self.logger_object.log(self.file_object, 'Pre-Processing - Successfully saved the pre-processed file at desired location.')

            return ("Pre-Processing Success")

        except Exception as e:
            self.logger_object.log(self.file_object,'Exception occured in data pre-processing. Exception message:  '+str(e))
            return "Error during input file pre-processing!Please check logs for details."

