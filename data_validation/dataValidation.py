import pandas as pd
from app_logger import logger


class DataValidation:
    """
        This class shall be used to validate the Prediction & Train data before Pre-processing.
    """
    def __init__(self,filepath_val,process_type):
         self.logger_object = logger.App_Logger()

    def data_validation(self, filepath_val,process_type):
        """
            This method does Data validation 1) No of Columns 2) Datatype for values
        """

        try:
            f = open("logs/DataValidation_Log.txt", 'a+')
            self.logger_object.log(f, "Data Validation - Column Length Validation Started!")

            csv = pd.read_csv(filepath_val)
            self.logger_object.log(f, "Data Validation - File read successfully!")
            if process_type == "P":
                if csv.shape[1] == 179:
                    if csv.iloc[:, 1:178].dtypes.unique() == 'int64':
                        csv.to_csv("Prediction_Files/Good_Raw/Pred_file.csv", index=None, header=True)
                        self.logger_object.log(f, "Data Validation - File Validated & Moved to Good Raw Folder")
                    else:
                        csv.to_csv("Prediction_Files/Bad_Raw/Pred_file.csv", index=None, header=True)
                        self.logger_object.log(f,"Data Validation - Invalid data type for column! File moved to Bad Raw Folder")
                        return "Error - File format is incorrect! Data type of features should be Int64. Please check once!"
                else:
                    csv.to_csv("Prediction_Files/Bad_Raw/Pred_file.csv", index=None, header=True)
                    self.logger_object.log(f,"Data Validation - Invalid Column Length for the file! File moved to Bad Raw Folder")
                    return "Error - File format is incorrect! No of columns should be 179. Please check once!"
            else:
                if process_type == "T":
                    if csv.shape[1] == 180:
                        if csv.iloc[:, 1:179].dtypes.unique() == 'int64':
                            csv.to_csv("Training_Files/Good_Raw/Train_file.csv", index=None, header=True)
                            self.logger_object.log(f, "Data Validation - File Validated & Moved to Good Raw Folder")
                        else:
                            csv.to_csv("Training_Files/Bad_Raw/Train_file.csv", index=None, header=True)
                            self.logger_object.log(f, "Data Validation - Invalid data type for column! File moved to Bad Raw Folder")
                            return "Error - File format is incorrect! Data type of features should be Int64. Please check once!"
                    else:
                        csv.to_csv("Training_Files/Bad_Raw/Train_file.csv", index=None, header=True)
                        self.logger_object.log(f, "Data Validation - Invalid Column Length for the file! File moved to Bad Raw Folder")
                        return "Error - File format is incorrect! No of columns should be 179. Please check once!"

            self.logger_object.log(f, "Data Validation Completed!")
            return "Validation Success"

            f.close()

        except OSError:
            f = open("logs/DataValidation_Log.txt", 'a+')
            self.logger_object.log(f, "Data Validation - Error occured while moving the file :: %s" % OSError)
            f.close()
            return "Error during input file File Validation!Please check logs for details."

        except Exception as e:
            f = open("logs/DataValidation_Log.txt", 'a+')
            self.logger_object.log(f, "Data Validation - Error Occured:: %s" % e)
            f.close()
            return "Error during input file validation!Please check logs for details."


