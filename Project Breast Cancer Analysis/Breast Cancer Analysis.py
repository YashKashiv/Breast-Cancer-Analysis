import sys

from PyQt5 import QtCore
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QLabel, QPushButton, QFileDialog, QMessageBox
from PyQt5.QtGui import QPixmap
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

class BreastCancerApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Breast Cancer Detection and Prevention")
        self.setGeometry(100, 100, 600, 400)

        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)

        layout = QVBoxLayout()

        self.label = QLabel("Upload your medical report for Breast Cancer Detection")
        layout.addWidget(self.label)

        self.upload_button = QPushButton("Upload Medical Report")
        self.upload_button.clicked.connect(self.loadDataset)
        layout.addWidget(self.upload_button)

        self.detect_button = QPushButton("Detect Cancer")
        self.detect_button.setEnabled(False)
        self.detect_button.clicked.connect(self.detectCancer)
        layout.addWidget(self.detect_button)

        self.image_label = QLabel(self)
        layout.addWidget(self.image_label)

        self.central_widget.setLayout(layout)

        self.dataset_loaded = False
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.model = None

        self.loadImage()

    def loadImage(self):
        pixmap = QPixmap("img.jpg")
        self.image_label.setPixmap(pixmap)
        self.image_label.setAlignment(QtCore.Qt.AlignCenter)

    def loadDataset(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        file_name, _ = QFileDialog.getOpenFileName(self, "Open Dataset", "", "CSV Files (*.csv);;All Files (*)", options=options)

        if file_name:
            try:
                data = load_breast_cancer()
                X, y = data.data, data.target

                self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                self.label.setText("Report Loaded. Click 'Detect Cancer' to analyze.")
                self.detect_button.setEnabled(True)
                self.dataset_loaded = True
            except Exception as e:
                self.showMessageBox("Error", "An error occurred while loading the dataset.")

    def detectCancer(self):
        if self.dataset_loaded:
            try:
                self.model = RandomForestClassifier(random_state=42)
                self.model.fit(self.X_train, self.y_train)

                y_pred = self.model.predict(self.X_test)
                accuracy = accuracy_score(self.y_test, y_pred)

                self.showMessageBox("Result", f"Breast Cancer Detection Accuracy: {accuracy:.2f}")
            except Exception as e:
                self.showMessageBox("Error", "An error occurred while detecting breast cancer.")
        else:
            self.showMessageBox("Error", "Please load a dataset before trying to detect cancer.")

    def showMessageBox(self, title, message):
        msg_box = QMessageBox(self)
        msg_box.setWindowTitle(title)
        msg_box.setText(message)
        msg_box.exec_()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = BreastCancerApp()
    window.show()
    sys.exit(app.exec_())
