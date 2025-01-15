import os
import json
from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout, QLineEdit, QPushButton, QFileDialog, QMessageBox

class SaveLoadPlotHandler(QWidget):
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
        file_path, test = QFileDialog.getOpenFileName(self, "Load Data", load_dir, f"Plot Data Files (*{file_type})")
        print(test)

        data_dict = None

        if file_path:
            # Read the data from the file and load it as a dictionary
            with open(file_path, 'r') as file:
                data_dict = json.load(file)

        
        return data_dict


class SaveLoadResHandler(QWidget):
    def __init__(self):
        super().__init__()

    def save_data(self):
        if not os.path.exists('problem_description.json'):
            QMessageBox.warning(self, "Warning", "No data to save!")
            return

        # Define the save directory and ensure it exists
        save_dir = os.path.join(os.getcwd(), "saved_data")
        os.makedirs(save_dir, exist_ok=True)

        # Define file path
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Data", save_dir, f"Result Data Files (*.firOres)")

        if file_path:
            # Ensure file has the correct extension
            if not file_path.endswith(".firOres"):
                file_path += ".firOres"

            # Write the dictionary to the file in JSON format
            res_file_paths = {
                "result_valid": "result_valid.json",
                "result_invalid": "result_invalid.json",
                "problem_description": "problem_description.json",
            }

            combined_data = {}

            for key, res_file_path in res_file_paths.items():
                if os.path.exists(res_file_path):
                    with open(res_file_path, 'r') as f:
                        combined_data[key] = json.load(f)  # Read and parse each JSON file
                else:
                    combined_data[key] = None

            with open(file_path, 'w') as f:
                json.dump(combined_data, f, indent=4)  # Write combined JSON to file

            QMessageBox.information(self, "Success", f"Data saved successfully to {file_path}")

    def load_data(self):
        
        # Define the load directory
        load_dir = os.path.join(os.getcwd(),  "saved_data")

        # Open a file dialog to select the .plotdata file
        file_path, test = QFileDialog.getOpenFileName(self, "Load Data", load_dir, f"Plot Data Files (*.firOres)")

        if os.path.exists(file_path):  # Check if the combined file exists
            with open(file_path, 'r') as f:
                try:
                    combined_data = json.load(f)  # Load the combined JSON file
                except json.JSONDecodeError:
                    print(f"Error: {file_path} is not a valid JSON file.")
                    return

            for key, content in combined_data.items():
                if content is None:
                    continue
                output_file = f"{key}.json"
                with open(output_file, 'w') as f:
                    json.dump(content, f, indent=4)  # Write each part to its own file
        else:
            print(f"Error: {file_path} does not exist.")

        
