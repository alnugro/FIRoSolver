import sys
from PyQt6.QtWidgets import QApplication, QMainWindow, QPushButton, QMessageBox


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # Set up the main window
        self.setWindowTitle("PyQt6 Pop-up Example")
        self.setGeometry(100, 100, 300, 200)

        # Create buttons
        self.yes_no_button = QPushButton("Ask Yes/No", self)
        self.yes_no_button.setGeometry(50, 50, 200, 40)
        self.yes_no_button.clicked.connect(self.show_yes_no_dialog)

        self.error_button = QPushButton("Show Error", self)
        self.error_button.setGeometry(50, 120, 200, 40)
        self.error_button.clicked.connect(self.show_error_dialog)

    def show_yes_no_dialog(self):
        # Create a QMessageBox for Yes/No prompt
        reply = QMessageBox.question(self, 'Confirmation', 'Do you want to proceed?',
                                     QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel)

        if reply == QMessageBox.StandardButton.Yes:
            print("User selected Yes")
        else:
            print("User selected No")

    def show_error_dialog(self):
        # Create a QMessageBox for exception or error
        error_msg = QMessageBox()
        error_msg.setIcon(QMessageBox.Icon.Critical)
        error_msg.setWindowTitle("Error")
        error_msg.setText("An exception occurred!")
        error_msg.setStandardButtons(QMessageBox.StandardButton.Ok)

        error_msg.exec()  # Display the pop-up dialog


if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec())
