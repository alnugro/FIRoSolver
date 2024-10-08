import os
import json
from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout, QLineEdit, QPushButton, QFileDialog, QMessageBox

class SaveLoadHandler(QWidget):
    def __init__(self):
        super().__init__()



    def save_data(self, data_dict : dict, plot_flag):
        if plot_flag:
            file_type = ".firOplt"
        else:
            file_type = ".firOres"

        
        if not data_dict:
            QMessageBox.warning(self, "Warning", "No data to save!")
            return

        # Define the save directory and ensure it exists
        save_dir = os.path.join(os.getcwd(), "saved_data")
        os.makedirs(save_dir, exist_ok=True)

        # Define file path
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Data", save_dir, f"Plot Data Files (*{file_type})")

        if file_path:
            # Ensure file has the correct extension
            if not file_path.endswith(file_type):
                file_path += file_type

            # Write the dictionary to the file in JSON format
            with open(file_path, 'w') as file:
                json.dump(data_dict, file)
            QMessageBox.information(self, "Success", "Data saved successfully!")

    def load_data(self, plot_flag):
        if plot_flag:
            file_type = ".firOplt"
        else:
            file_type = ".firOres"

        # Define the load directory
        load_dir = os.path.join(os.getcwd(),  "saved_data")

        # Open a file dialog to select the .plotdata file
        file_path, _ = QFileDialog.getOpenFileName(self, "Load Data", load_dir, f"Plot Data Files (*{file_type})")

        data_dict = None

        if file_path:
            # Read the data from the file and load it as a dictionary
            with open(file_path, 'r') as file:
                data_dict = json.load(file)

        
        return data_dict


