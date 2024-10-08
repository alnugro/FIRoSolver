import os
import json
from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout, QLineEdit, QPushButton, QFileDialog, QMessageBox

class PlotDataApp(QWidget):
    def __init__(self):
        super().__init__()

        # Setup UI
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Save and Load Plot Data")

        # Create layout
        layout = QVBoxLayout()

        # Create QLineEdit for data input
        self.data_input = QLineEdit(self)
        layout.addWidget(self.data_input)

        # Create Save button
        self.save_button = QPushButton('Save Data', self)
        self.save_button.clicked.connect(self.save_data)
        layout.addWidget(self.save_button)

        # Create Load button
        self.load_button = QPushButton('Load Data', self)
        self.load_button.clicked.connect(self.load_data)
        layout.addWidget(self.load_button)

        # Set the layout to the main window
        self.setLayout(layout)

    def save_data(self):
        # Get the data from QLineEdit
        data = self.data_input.text()

        # Ensure there's something to save
        if not data:
            QMessageBox.warning(self, "Warning", "No data to save!")
            return

        # Create a dictionary to store the data
        data_dict = {
            "plot_data": data
        }

        # Define the save directory and ensure it exists
        save_dir = os.path.join(os.getcwd(), "currentproject", "saved_data")
        os.makedirs(save_dir, exist_ok=True)

        # Define file path
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Data", save_dir, "Plot Data Files (*.plotdata)")

        if file_path:
            # Ensure file has the correct extension
            if not file_path.endswith(".plotdata"):
                file_path += ".plotdata"

            # Write the dictionary to the file in JSON format
            with open(file_path, 'w') as file:
                json.dump(data_dict, file)
            QMessageBox.information(self, "Success", "Data saved successfully!")

    def load_data(self):
        # Define the load directory
        load_dir = os.path.join(os.getcwd(), "currentproject", "saved_data")

        # Open a file dialog to select the .plotdata file
        file_path, _ = QFileDialog.getOpenFileName(self, "Load Data", load_dir, "Plot Data Files (*.plotdata)")

        if file_path:
            # Read the data from the file and load it as a dictionary
            with open(file_path, 'r') as file:
                data_dict = json.load(file)

            # Set the data into the QLineEdit if the key exists
            if "plot_data" in data_dict:
                self.data_input.setText(data_dict["plot_data"])
                QMessageBox.information(self, "Success", "Data loaded successfully!")
            else:
                QMessageBox.warning(self, "Warning", "No valid plot data found in the file.")

# Run the application
if __name__ == '__main__':
    app = QApplication([])
    window = PlotDataApp()
    window.show()
    app.exec()
