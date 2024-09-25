from PyQt6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QPushButton, QSlider, QComboBox, QSpinBox, QTextEdit, QTableWidget, QTableWidgetItem, QWidget, QFrame
from PyQt6.QtCore import Qt


class LiveLogger():
    def __init__(self, main_window):
        # Store reference to the main window
        self.main_window = main_window
        self.loggerline = 0

    def plog(self, text):
            self.main_window.logger_box.append(f"[{self.loggerline}] {text}")
            self.loggerline += 1