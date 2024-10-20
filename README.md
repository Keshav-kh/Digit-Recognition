
## 🎨 Digit Recognition Drawing App 🖌️

Welcome to the **Digit Recognition Drawing App**! This project allows users to draw digits on a 28x28 canvas, and the AI model predicts the digit using a neural network (MLP Classifier). The app is built using `Tkinter` for the GUI and `scikit-learn` for the machine learning model.

---

### ✨ Features
- 🎨 **Interactive Drawing Canvas**: Draw digits freely on a 28x28 grid.
- 🤖 **AI-powered Prediction**: Uses a neural network (MLP) for better accuracy.
- 🧹 **Clean Board**: Reset the canvas with a single click.
- 📈 **Pre-trained Model Loading**: Avoid re-training with a pre-saved model.

---

### 🛠️ Prerequisites
- Python 3.x
- Required libraries:
  - `numpy`
  - `matplotlib`
  - `scikit-learn`
  - `tkinter` (Usually pre-installed with Python)
  - `joblib`

Install the required libraries using:

```bash
pip install numpy matplotlib scikit-learn joblib
```

---

### 📦 Files Included
- `main.py` - The main Python file to run the app.
- `mnist_mlp_model.pkl` - Pre-trained MLP model (auto-generated after training).

---

### 🚀 How to Run the App
1. **Clone the Repository:**
    ```bash
    git clone https://github.com/keshav-kh/digit-recognition-app.git
    ```
2. **Navigate to the Project Directory:**
    ```bash
    cd digit-recognition-app
    ```
3. **Run the Application:**
    ```bash
    python main.py
    ```

After the application starts, draw a digit on the canvas, and click **"Predict Digit"**. The AI will predict the digit and display it. Use **"Clean Board"** to reset the canvas.

---

### 📊 Model Training
The application uses the **MNIST dataset** and an **MLP Classifier** from `scikit-learn`. The model is trained once and saved as `mnist_mlp_model.pkl` to skip re-training every time the app is run.

If you wish to re-train the model, delete the `.pkl` file, and the app will automatically train a new model.

---

### 🤝 Contribution
Feel free to open issues or submit pull requests. Any improvements to the model or the app's user interface are welcome!

---

### 📝 License
This project is open-source and available under the MIT License.

---

**Have fun drawing! 🖌️✨**
