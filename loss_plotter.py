# Track changes in loss and learning rate during execution

import matplotlib
import matplotlib.pyplot as plt
import os
import time
import json

class LossPlotter:
    def __init__(self, log_file, update_interval=1.0):
        self.log_file = log_file
        self.update_interval = update_interval
        self.running = True
        
        # Use non-interactive backend
        matplotlib.use('Agg')
        
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        self.epochs = []
        self.losses = []
        self.lr_values = []
        
    def update_plot(self):
        if os.path.exists(self.log_file):
            try:
                with open(self.log_file, 'r') as f:
                    data = [json.loads(line) for line in f if line.strip()]
                    
                if not data:
                    return
                    
                self.epochs = [entry.get('epoch', 0) for entry in data]
                self.losses = [entry.get('loss', 0) for entry in data]
                self.lr_values = [entry.get('lr', 0) for entry in data]
                
                # Clear the axes and redraw
                self.ax.clear()
                # Plot loss
                self.ax.plot(self.epochs, self.losses, 'b-', label='Training Loss')
                self.ax.set_xlabel('Epoch')
                self.ax.set_ylabel('Loss', color='b')
                self.ax.tick_params(axis='y', labelcolor='b')
                
                # Add learning rate on secondary y-axis
                if any(self.lr_values):
                    ax2 = self.ax.twinx()
                    ax2.plot(self.epochs, self.lr_values, 'r-', label='Learning Rate')
                    ax2.set_ylabel('Learning Rate', color='r')
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
        """Method for non-interactive plotting to run in thread"""
        print("Starting non-interactive plotting in background")
        while self.running:
            self.update_plot()
            time.sleep(self.update_interval)
    
    def stop_plotting(self):
        self.running = False
        plt.close(self.fig)
        print("Plotting stopped")
