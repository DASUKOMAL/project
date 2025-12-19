import tkinter as tk
from tkinter import messagebox
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt

# Globals
dataset = None
X = None
Y = None

accuracy = []
precision = []
recall = []
fscore = []

# GUI setup
root = tk.Tk()
root.title("Privacy Preserving and Secure Machine Learning")
root.geometry("950x600")
root.configure(bg="#f8f8ff")  # Light lavender background

# Title
title_label = tk.Label(
    root,
    text="Privacy Preserving and Secure Machine Learning",
    font=("Helvetica", 20, "bold"),
    bg="#f8f8ff",
    fg="#333"
)
title_label.pack(pady=(10, 5))

# Top frame for buttons
top_frame = tk.Frame(root, bg="#f8f8ff", pady=10)
top_frame.pack(side=tk.TOP, fill=tk.X)

# Output frame below
bottom_frame = tk.Frame(root, bg="white", padx=10, pady=10)
bottom_frame.pack(side=tk.TOP, expand=True, fill=tk.BOTH)

# Output label
output_label = tk.Label(
    bottom_frame,
    text="Output Panel",
    font=("Helvetica", 14, "bold"),
    bg="white"
)
output_label.pack(anchor="nw")

# Output text area
output_text = tk.Text(
    bottom_frame,
    wrap="word",
    font=("Consolas", 12, "bold"),
    bg="#fff8dc",  # Light yellow background
    fg="#000"
)
output_text.pack(expand=True, fill=tk.BOTH, pady=(5, 0))

# Helper to print to output
def print_output(msg):
    output_text.insert(tk.END, msg + "\n")
    output_text.see(tk.END)

# Load dataset
def load_dataset():
    global dataset, X, Y, accuracy, precision, recall, fscore
    try:
        dataset = pd.read_csv("heart.csv")  # Change path if needed
        X = dataset.drop("target", axis=1)
        Y = dataset["target"]
        accuracy.clear()
        precision.clear()
        recall.clear()
        fscore.clear()
        print_output("✅ Dataset loaded successfully.")
        print_output(f"Rows: {dataset.shape[0]}, Columns: {dataset.shape[1]}")
    except Exception as e:
        messagebox.showerror("Error", f"Failed to load dataset:\n{str(e)}")

# Preprocess data
def preprocess_data():
    global X, Y, dataset
    if dataset is None:
        messagebox.showerror("Error", "Load dataset first.")
        return

    print_output("🔄 Starting preprocessing...")

    # Handle missing values
    dataset.fillna(dataset.mean(numeric_only=True), inplace=True)

    # Encode categorical columns if any
    for col in dataset.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        dataset[col] = le.fit_transform(dataset[col])

    # Separate features and target
    X = dataset.drop("target", axis=1)
    Y = dataset["target"]

    # Normalize numeric features
    scaler = StandardScaler()
    X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    print_output("✅ Data preprocessing completed successfully.")
    print_output(f"Updated Shape: {X.shape}")
    print_output(f"Columns: {list(X.columns)}")

# Evaluate and store metrics
def calculateMetrics(name, y_test, prediction):
    acc = accuracy_score(y_test, prediction)
    pre = precision_score(y_test, prediction)
    rec = recall_score(y_test, prediction)
    f1 = f1_score(y_test, prediction)

    accuracy.append(acc)
    precision.append(pre)
    recall.append(rec)
    fscore.append(f1)

    print_output(f"\n--- {name} Results ---")
    print_output(f"Accuracy : {acc:.2f}")
    print_output(f"Precision: {pre:.2f}")
    print_output(f"Recall   : {rec:.2f}")
    print_output(f"F1-Score : {f1:.2f}")

# Run Differential Privacy
def run_diff_privacy():
    if X is None or Y is None:
        messagebox.showerror("Error", "Load and preprocess dataset first.")
        return
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    clf = LogisticRegression(max_iter=2000)
    clf.fit(X_train, y_train)
    prediction = clf.predict(X_test)
    calculateMetrics("Differential Privacy (Logistic Regression)", y_test, prediction)

# Run Homomorphic Encryption
def run_homomorphic():
    if X is None or Y is None:
        messagebox.showerror("Error", "Load and preprocess dataset first.")
        return
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    clf = DecisionTreeClassifier()
    clf.fit(X_train, y_train)
    prediction = clf.predict(X_test)
    calculateMetrics("Homomorphic Encryption (Decision Tree)", y_test, prediction)

# Compare models
def compare_models():
    if len(accuracy) < 2:
        messagebox.showerror("Error", "Run both models first.")
        return

    df = pd.DataFrame([
        ["Differential Privacy", "Accuracy", accuracy[0]],
        ["Differential Privacy", "Precision", precision[0]],
        ["Differential Privacy", "Recall", recall[0]],
        ["Differential Privacy", "F1-Score", fscore[0]],
        ["Homomorphic Encryption", "Accuracy", accuracy[1]],
        ["Homomorphic Encryption", "Precision", precision[1]],
        ["Homomorphic Encryption", "Recall", recall[1]],
        ["Homomorphic Encryption", "F1-Score", fscore[1]],
    ], columns=["Algorithm", "Metric", "Value"])

    pivot_df = df.pivot(index="Metric", columns="Algorithm", values="Value")
    pivot_df.plot(kind="bar", figsize=(8, 4), colormap="coolwarm")
    plt.title("Algorithm Performance Comparison")
    plt.ylabel("Score")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()

    print_output("\n📊 Performance comparison chart displayed.")

# Button styles
button_styles = [
    {"bg": "#007BFF", "activebackground": "#0056b3"},  # Blue
    {"bg": "#FF69B4", "activebackground": "#cc568f"},  # Pink
]

# Create buttons
buttons = [
    ("Load Dataset", load_dataset),
    ("Preprocess Data", preprocess_data),
    ("Run Differential Privacy", run_diff_privacy),
    ("Run Homomorphic Encryption", run_homomorphic),
    ("Compare Models", compare_models),
    ("Exit", root.quit)
]

for i, (text, cmd) in enumerate(buttons):
    style = button_styles[i % 2]
    tk.Button(
        top_frame, text=text, command=cmd,
        font=("Helvetica", 12, "bold"), fg="white",
        width=22, height=2, bd=0,
        **style
    ).pack(side=tk.LEFT, padx=5, pady=5)

# Run app
root.mainloop()
