import sys
from PyQt6.QtWidgets import QApplication, QMainWindow, QTextEdit, QVBoxLayout, QPushButton, QWidget
from PyQt6.QtCore import QThread, pyqtSignal, Qt
from time import sleep


# Define the backend worker thread class
class SolverThread(QThread):
    log_message = pyqtSignal(str)
    presolve_done = pyqtSignal(tuple)

    def __init__(self, solver_input):
        super().__init__()
        self.solver_input = solver_input

    def run(self):
        # Simulating backend work
        for i in range(5):
            self.log_message.emit(f"Running backend process step {i + 1}")
            sleep(1)  # Simulate a time-consuming task
        # Emit completion signal
        self.presolve_done.emit(("Success", "No issues"))


# Define the LiveLogger class
class LiveLogger:
    def __init__(self, main_window):
        self.main_window = main_window
        self.loggerline = 0

    def plog(self, text):
        self.main_window.logger_box.append(f"[{self.loggerline}] {text}")
        self.loggerline += 1


# Define the MainWindow class
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # Initialize UI components
        self.init_ui()

        # Initialize logger
        self.logger = LiveLogger(self)

    def init_ui(self):
        # Create a central widget and layout
        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)
        layout = QVBoxLayout(self.central_widget)

        # Create a QTextEdit for logging
        self.logger_box = QTextEdit(self)
        self.logger_box.setReadOnly(True)
        layout.addWidget(self.logger_box)

        # Create a button to start the solver
        self.start_button = QPushButton("Start Solver", self)
        self.start_button.clicked.connect(self.start_solver)
        layout.addWidget(self.start_button)

        # Set window title and size
        self.setWindowTitle("Solver with LiveLogger")
        self.setGeometry(100, 100, 400, 300)

    def start_solver(self):
        # Create dummy solver input
        initial_solver_input = {"gurobi_thread": 1}

        # Create the solver thread
        self.solver_thread = SolverThread(initial_solver_input)

        # Connect signals to slots
        self.solver_thread.log_message.connect(self.logger.plog)
        self.solver_thread.presolve_done.connect(self.on_presolve_done)

        # Start the solver thread
        self.logger.plog("Starting the solver...")
        self.solver_thread.start()

    def on_presolve_done(self, results):
        self.logger.plog(f"Solver finished with results: {results}")


# Application entry point
if __name__ == "__main__":
    app = QApplication(sys.argv)

    # Create and show the main window
    main_window = MainWindow()
    main_window.show()

    sys.exit(app.exec())
