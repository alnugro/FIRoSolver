import sys
from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QPushButton, QTableWidget, 
    QTableWidgetItem, QAbstractItemView
)
from PyQt6.QtCore import Qt

class DynamicTableWidget(QWidget):
    def __init__(self):
        super().__init__()

        self.initUI()

    def initUI(self):
        self.setWindowTitle('Dynamic Table with Buttons')
        self.setGeometry(100, 100, 500, 400)

        # Layout for the window
        layout = QVBoxLayout()

        # Create the table
        self.table = QTableWidget(0, 2)  # Initially no rows, two columns
        self.table.setHorizontalHeaderLabels(['Index', 'Button'])
        
        # Disable editing
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        
        layout.addWidget(self.table)

        # Add row button
        self.addRowButton = QPushButton('Add Row')
        self.addRowButton.clicked.connect(self.addRow)
        layout.addWidget(self.addRowButton)

        self.setLayout(layout)

    def addRow(self):
        # Get the current row count
        rowCount = self.table.rowCount()

        # Insert a new row
        self.table.insertRow(rowCount)

        # Add row number in the first column
        self.table.setItem(rowCount, 0, QTableWidgetItem(str(rowCount + 1)))

        # Create a button in the second column
        button = QPushButton(f'Button {rowCount + 1}')
        button.clicked.connect(lambda _, row=rowCount + 1: self.buttonClicked(row))

        # Add the button to the table
        self.table.setCellWidget(rowCount, 1, button)

    def buttonClicked(self, row):
        print(f'Button clicked for row {row}')

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = DynamicTableWidget()
    window.show()
    sys.exit(app.exec())
