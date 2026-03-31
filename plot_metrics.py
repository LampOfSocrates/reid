import argparse
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

def is_notebook():
    """
    Detects if the code is running inside a Jupyter notebook environment.
    """
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreter

def plot_metrics(csv_path, output_png="training_metrics.png"):
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return

    # 1. Read the metrics CSV
    df = pd.parse_csv(csv_path) if hasattr(pd, "parse_csv") else pd.read_csv(csv_path)
    
    # 2. Plot the data
    plt.figure(figsize=(10, 6))
    
    # We plot the Loss over Epochs
    if 'Epoch' in df.columns and 'Loss' in df.columns:
        plt.plot(df['Epoch'], df['Loss'], marker='o', linestyle='-', color='b', label='Training Loss')
        plt.title('Training Loss over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Average Loss')
    else:
        print("CSV doesn't contain 'Epoch' and 'Loss' columns.")
        return
        
    plt.grid(True)
    plt.legend()
    
    # 3. Detect notebook and render
    if is_notebook():
        print("Detected Jupyter Notebook environment. Rendering plot on screen...")
        plt.show()
    else:
        plt.savefig(output_png)
        print(f"Plot saved to {output_png}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Plot Training Metrics")
    parser.add_argument('--csv', type=str, default='metrics.csv', help='Path to the metrics CSV file')
    parser.add_argument('--output', type=str, default='training_metrics.png', help='Path to save output PNG')
    args = parser.parse_args()
    
    plot_metrics(args.csv, args.output)
