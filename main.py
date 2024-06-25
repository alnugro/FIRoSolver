import sys
from PyQt6.QtWidgets import QApplication
from GUI.main_window import MainWindow
from backend.solver_init import SolverInit

def run_main_window():
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec())



run_main_window()
#solver=SolverInit()
