import os
from werkzeug.utils import secure_filename
from app_logger import logger

os.putenv('LANG', 'en_US.UTF-8')
os.putenv('LC_ALL', 'en_US.UTF-8')

class UploadFile:
    """
        This class shall be used to upload the file provided for Prediction or Re-Training.
    """

    def __init__(self,file):
        self.file_object = open("logs/UploadFile_Log.txt", 'a+')
        self.logger_object = logger.App_Logger()

    def upload_file(self,file):

        try:

            if file.filename == '':
                self.logger_object.log(self.file_object, 'Upload File - No file selected for uploading')
                return "No file selected for uploading"
            if file:
                filename = secure_filename(file.filename)
                file.save(os.path.join("Input_Files", filename))
                self.logger_object.log(self.file_object, 'Upload File - File uploaded at the desired location')
                return "File successfully uploaded at the desired location"
            else:
                self.logger_object.log(self.file_object, 'Upload File - File not found!')
                return "File not found"

        except Exception as e:
            return "Error during file upload!Please check logs for details."
