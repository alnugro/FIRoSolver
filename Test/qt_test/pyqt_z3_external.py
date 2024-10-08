from PyQt6.QtCore import QProcess, QByteArray
from PyQt6.QtWidgets import QApplication, QMainWindow, QPushButton
import sys

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Z3 in PyQt")
        self.button = QPushButton("Run Z3", self)
        self.button.clicked.connect(self.run_z3)
        self.setCentralWidget(self.button)
        self.process = QProcess()
        self.process.readyReadStandardOutput.connect(self.handle_stdout)
        self.process.finished.connect(self.process_finished)

    def run_z3(self):
        self.process.start("python", ["z3_script.py"])

    def handle_stdout(self):
        data = self.process.readAllStandardOutput()
        stdout = bytes(data).decode('utf-8')
        print(stdout)

    def process_finished(self):
        print("Process finished")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
