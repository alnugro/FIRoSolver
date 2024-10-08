import sys
from PyQt6.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget, QLineEdit

class SecondWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Second Window")
        self.setGeometry(200, 200, 300, 200)
        
        # Create a layout and a text field
        layout = QVBoxLayout()
        self.text_field = QLineEdit(self)
        layout.addWidget(self.text_field)
        
        self.setLayout(layout)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Main Window")
        self.setGeometry(100, 100, 400, 300)

        # Create a central widget and set layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        layout = QVBoxLayout()

        # Create a button that opens the second window
        self.button = QPushButton("Open Second Window")
        self.button.clicked.connect(self.open_second_window)
        layout.addWidget(self.button)
        
        central_widget.setLayout(layout)

    def open_second_window(self):
        # Create and show the second window with the text field
        self.second_window = SecondWindow()
        self.second_window.show()

# Main execution
if __name__ == '__main__':
    app = QApplication(sys.argv)
    
    main_window = MainWindow()
    main_window.show()
    
    sys.exit(app.exec())
