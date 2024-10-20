from sklearn.datasets import fetch_openml
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import tkinter as tk
import joblib

def train_or_load_model():
    try:
        # Load the pre-trained model and scaler
        clf, scaler = joblib.load("mnist_mlp_model.pkl")
    except:
        # Train the model and save it if it doesn't exist
        mnist = fetch_openml("mnist_784")
        x, y = mnist["data"], mnist["target"]

        # Normalize the input data
        scaler = MinMaxScaler()
        x_train = scaler.fit_transform(x)

        y_train = np.array(y).astype(np.int8)
        shuffle_index = np.random.permutation(60000)
        x_train, y_train = x_train[shuffle_index], y_train[shuffle_index]

        # Use MLP Classifier (Neural Network)
        clf = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=300, random_state=42)
        clf.fit(x_train, y_train)

        # Save the model and scaler together
        joblib.dump((clf, scaler), "mnist_mlp_model.pkl")
        
    return clf, scaler

class DrawingApp:
    def __init__(self, root, clf, scaler):
        self.root = root
        self.root.title("Digit Recognition Drawing App")
        self.clf = clf
        self.scaler = scaler

        self.canvas = tk.Canvas(root, width=280, height=280, bg="white")
        self.canvas.pack()

        self.draw_grid()

        self.canvas.bind("<B1-Motion>", self.paint)
        
        self.get_matrix_button = tk.Button(root, text="Predict Digit", command=self.get_matrix)
        self.get_matrix_button.pack()

        self.clear_button = tk.Button(root, text="Clean Board", command=self.clear_canvas)
        self.clear_button.pack()

        self.result_label = tk.Label(root, text="Draw a digit and click 'Predict Digit'")
        self.result_label.pack()

        self.filled_pixels = np.zeros((28, 28), dtype=np.uint8)

    def draw_grid(self):
        for i in range(29):
            self.canvas.create_line(i * 10, 0, i * 10, 280, fill="lightgray")
            self.canvas.create_line(0, i * 10, 280, i * 10, fill="lightgray")

    def paint(self, event):
        x, y = event.x, event.y
        col = x // 10
        row = y // 10
        self.canvas.create_rectangle(col * 10, row * 10, (col + 1) * 10, (row + 1) * 10, fill="black", outline="black")
        self.filled_pixels[row, col] = 1

    def get_matrix(self):
        matrix = self.filled_pixels * 255
        some_digit_flattened = matrix.ravel().reshape(1, -1)
        
        some_digit_flattened = self.scaler.transform(some_digit_flattened)

        prediction = self.clf.predict(some_digit_flattened)
        
        self.result_label.config(text=f"Predicted Digit: {prediction[0]}")
        
        plt.imshow(matrix, cmap=matplotlib.cm.binary, interpolation='nearest')
        plt.show()

    def clear_canvas(self):
        self.canvas.delete("all")
        self.draw_grid()
        self.filled_pixels = np.zeros((28, 28), dtype=np.uint8)

if __name__ == "__main__":
    clf, scaler = train_or_load_model()
    root = tk.Tk()
    app = DrawingApp(root, clf, scaler)
    root.mainloop()
