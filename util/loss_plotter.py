# Track changes in loss and learning rate during execution

import matplotlib
import matplotlib.pyplot as plt
import os
import time
import json

class LossPlotter:
    def __init__(self, log_file, update_interval=1.0, left_key='loss', right_key='lr', left_label='Loss', right_label='Learning Rate'):
        self.log_file = log_file
        self.update_interval = update_interval
        self.running = True

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
                    
                #self.epochs = [entry.get('epoch', 0) for entry in data]
                self.steps = [entry.get('step', 0) for entry in data]
                #self.losses = [entry.get('loss', 0) for entry in data]
                #self.lr_values = [entry.get('lr', 0) for entry in data]
                
                self.left = [entry.get(self.left_key, 0) for entry in data]
                self.right = [entry.get(self.right_key, 0) for entry in data]

                # Clear the axes and redraw
                self.ax.clear()
                # Plot loss
                self.ax.plot(self.steps, self.left, 'b-', label=self.left_label)
                self.ax.set_xlabel('Step')
                self.ax.set_ylabel(self.left_label, color='b')
                self.ax.tick_params(axis='y', labelcolor='b')
                
                # Add learning rate on secondary y-axis
                if any(self.right):
                    ax2 = self.ax.twinx()
                    ax2.plot(self.steps, self.right, 'r-', label=self.right_label)
                    ax2.set_ylabel(self.right_label, color='r')
                    ax2.tick_params(axis='y', labelcolor='r')
                    ax2.legend(loc='upper right')
                
                # Add a title and legend
                self.ax.set_title('Training Progress')
                self.ax.legend(loc='upper left')
                
                # Adjust layout
                self.fig.tight_layout()
                
                # Save the current plot to disk
                self.fig.savefig(os.path.join(os.path.dirname(self.log_file), 'training_progress.png'))
            
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
        plt.close(self.fig)
        print("Plotting stopped")
