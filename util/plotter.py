# Track changes in loss and learning rate during execution

import matplotlib
import matplotlib.pyplot as plt
import os
import time
import json

class Plotter:
    def __init__(self, log_file, update_interval=1.0, left_key='loss', right_key='lr', left_label='Loss', right_label='Learning Rate', output_png='training_progress.png'):
        self.log_dir = os.path.dirname(log_file)

        self.log_file = log_file
        self.update_interval = update_interval
        self.running = True
        self.output_png = output_png

        matplotlib.use('Agg')
        
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        
        self.left_key = left_key
        self.right_key = right_key
        self.left_label = left_label
        self.right_label = right_label

    def update_plot(self):
        if os.path.exists(self.log_file):
            try:
                with open(self.log_file, 'r') as f:
                    data = [json.loads(line) for line in f if line.strip()]
                    
                if not data:
                    return
                    
                self.epochs = [entry.get('epoch', 0) for entry in data]
                self.left = [entry.get(self.left_key, 0) for entry in data]

                # For right axis (e.g., lr), only include points where right_key exists
                right_points = [(entry.get('epoch', 0), entry.get(self.right_key))
                                for entry in data if self.right_key in entry]
                if right_points:
                    right_epochs, right_values = zip(*right_points)
                else:
                    right_epochs, right_values = [], []

                # Clear axis
                self.ax.clear()
                
                # Plot both metrics on the same axis
                self.ax.plot(self.epochs, self.left, 'b-', label=self.left_label)
                if right_epochs:
                    self.ax.plot(right_epochs, right_values, 'r-', label=self.right_label)
                
                self.ax.set_xlabel('Epoch')
                self.ax.set_ylabel(self.left_label) # "Loss" as y-axis label
                self.ax.set_title('Training Progress')
                self.ax.legend(loc='upper left')
                self.fig.tight_layout()

                # Use the stored base directory instead of getting it from log_file
                if os.path.isabs(self.output_png) or os.path.dirname(self.output_png):
                    output_path = self.output_png
                else:
                    output_path = os.path.join(self.log_dir, self.output_png)

                self.fig.savefig(output_path)
            
            except (json.JSONDecodeError, ValueError) as e:
                print(f"Error parsing log file: {e}")
    
    def start_plotting(self):
        """Method for plotting to run in thread"""
        print("Starting plotting in background")
        while self.running:
            self.update_plot()
            time.sleep(self.update_interval)
    
    def stop_plotting(self):
        self.running = False
        self.update_plot()
        plt.close(self.fig)
        print("Plotting stopped")
