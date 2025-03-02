import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import pandas as pd
import mainScript

class FraudDetectionGUI:

    def __init__(self, root):
        self.root = root
        self.root.title("Fraud Detection System")
        self.fraud_detection = mainScript.FraudDetection()

        # Style Configuration
        style = ttk.Style()
        style.configure("TButton", font=("Arial", 12, "bold"), padding=10)
        style.configure("TLabel", font=("Arial", 16, "bold"), background="#f0f8ff", foreground="#333333")

        # Title Label
        title_label = ttk.Label(root, text="Fraud Detection System", style="TLabel")
        title_label.grid(row=0, column=0, columnspan=3, pady=20)

        # Buttons
        self.load_dataset_button = ttk.Button(root, text="Load Dataset", command=self.load_dataset)
        self.load_dataset_button.grid(row=1, column=1, padx=20, pady=10)

        self.predict_fraud_button = ttk.Button(root, text="Predict Fraud", command=self.predict_fraud)
        self.predict_fraud_button.grid(row=2, column=1, padx=20, pady=10)

        self.show_graphs_button = ttk.Button(root, text="Show Graphs", command=self.show_graphs)
        self.show_graphs_button.grid(row=3, column=1, padx=20, pady=10)

        # Frame for displaying data and graphs
        self.output_frame = ttk.Frame(root, borderwidth=2, relief="sunken")
        self.output_frame.grid(row=4, column=0, columnspan=3, padx=10, pady=20, sticky="nsew")

        self.output_text = tk.Text(self.output_frame, height=15, width=80, wrap="word")
        self.output_text.pack(side="left", fill="both", expand=True, padx=5, pady=5)

    def load_dataset(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if file_path:
            try:
                self.fraud_detection.dataset_path = file_path
                self.fraud_detection.loadData()
                self.output_text.delete(1.0, tk.END)
                self.output_text.insert(tk.END, f"Dataset loaded successfully from: {file_path}\n\n")
                self.output_text.insert(tk.END, self.fraud_detection.data.head().to_string())
                messagebox.showinfo("Success", "Dataset loaded successfully!")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load dataset: {str(e)}")

    def predict_fraud(self):
        try:
            self.fraud_detection.preprocessData()
            self.fraud_detection.loadModel()
            predictions = self.fraud_detection.predict_fraud(self.fraud_detection.data)
            self.fraud_detection.save_predictions(self.fraud_detection.data, predictions)
            self.output_text.delete(1.0, tk.END)
            self.output_text.insert(tk.END, "Fraud predictions saved successfully!\n\n")
            fraud_data = self.fraud_detection.data[self.fraud_detection.data['Prediction'] == 1]
            self.output_text.insert(tk.END, fraud_data.to_string())
            messagebox.showinfo("Success", "Fraud predictions saved successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to predict fraud: {str(e)}")

    def show_graphs(self):
        try:
            predictions_file = "predictions.csv"
            predictions_data = pd.read_csv(predictions_file)
            fig = plt.Figure(figsize=(6, 4), dpi=100)
            ax = fig.add_subplot(111)
            class_counts = predictions_data['Prediction'].value_counts()
            class_counts.plot(kind='bar', ax=ax, color=['skyblue', 'orange'])
            ax.set_title("Fraud vs Non-Fraud Predictions")
            ax.set_xlabel("Class")
            ax.set_ylabel("Count")

            # Display the graph in the GUI
            for widget in self.output_frame.winfo_children():
                widget.destroy()

            canvas = FigureCanvasTkAgg(fig, master=self.output_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(side="top", fill="both", expand=True)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to show graphs: {str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = FraudDetectionGUI(root)
    root.mainloop()
