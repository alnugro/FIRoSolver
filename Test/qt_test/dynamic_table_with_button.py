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
        self.setGeometry(100, 100, 800, 400)

        # Layout for the window
        layout = QVBoxLayout()

        # Create the table with 6 columns
        self.table = QTableWidget(0, 6)  # Initially no rows, six columns
        self.table.setHorizontalHeaderLabels(['Index', 'Data 1', 'Data 2', 'Button 1', 'Button 2', 'Button 3'])
        
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

        # Add placeholder data in the second and third columns
        self.table.setItem(rowCount, 1, QTableWidgetItem(f'Data {rowCount + 1} - 1'))
        self.table.setItem(rowCount, 2, QTableWidgetItem(f'Data {rowCount + 1} - 2'))

        # Create buttons in the fourth, fifth, and sixth columns
        button1 = QPushButton(f'Button 1 - Row {rowCount + 1}')
        button1.clicked.connect(lambda _, row=rowCount + 1: self.buttonClicked(row, 1))
        self.table.setCellWidget(rowCount, 3, button1)

        button2 = QPushButton(f'Button 2 - Row {rowCount + 1}')
        button2.clicked.connect(lambda _, row=rowCount + 1: self.buttonClicked(row, 2))
        self.table.setCellWidget(rowCount, 4, button2)

        button3 = QPushButton(f'Button 3 - Row {rowCount + 1}')
        button3.clicked.connect(lambda _, row=rowCount + 1: self.buttonClicked(row, 3))
        self.table.setCellWidget(rowCount, 5, button3)

    def buttonClicked(self, row, button_number):
        print(f'Button {button_number} clicked for row {row}')

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = DynamicTableWidget()
    window.show()
    sys.exit(app.exec())