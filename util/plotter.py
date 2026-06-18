# Track changes in loss and learning rate during execution
import argparse
import matplotlib
import matplotlib.pyplot as plt
import os
import time
import json
import tempfile
import shutil
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Train a text-conditional diffusion model for tile-based level generation")
    
    # Dataset args
    parser.add_argument("--log_file", type=str, default=None, help="The the filepath of the file to get the data from")
    parser.add_argument("--left_key", type=str, default=None, help="The key for the left y-axis")
    parser.add_argument("--right_key", type=str, default=None, help="The key for the right y-axis")
    parser.add_argument("--left_label", type=str, default=None, help="The label for the left y-axis")
    parser.add_argument("--right_label", type=str, default=None, help="The label for the right y-axis")
    parser.add_argument("--output_png", type=str, default="output.png", help="The output png file")
    parser.add_argument("--update_interval", type=int, default=1.0, help="The update inteval in epochs")
    parser.add_argument("--start_point", type=int, default=None, help="The start point for the plot")
    parser.add_argument("--x_key", type=str, default="epoch", help="The key to use for the x-axis ('epoch' or 'step')")

    return parser.parse_args()


def main():
    args = parse_args()

    log_file = args.log_file
    left_key = args.left_key
    right_key = args.right_key
    left_label = args.left_label
    right_label = args.right_label
    output_png = args.output_png
    update_interval = args.update_interval
    start_point = args.start_point
    x_key = args.x_key

    general_update_plot(log_file, left_key, right_key, left_label, right_label, output_png,
                        update_interval=update_interval, startPoint=start_point, x_key=x_key)


def _step_png_path(output_png):
    p = Path(output_png)
    return str(p.with_stem(p.stem + "_by_step"))


def general_update_plot(log_file, left_key, right_key, left_label, right_label, output_png,
                        update_interval=1.0, startPoint=None, x_key="epoch"):

    log_dir = os.path.dirname(log_file)
    
    # Create figure here and ensure it's closed
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)
    
    try:
        if os.path.exists(log_file):
            with open(log_file, 'r') as f:
                data = [json.loads(line) for line in f if line.strip()]
            
            if not data:
                return
            
            # Filter to entries that have the requested x_key
            data = [entry for entry in data if x_key in entry]

            if startPoint is not None:
                data = [entry for entry in data if entry.get(x_key, 0) >= startPoint]
            
            if not data:
                return

            x_values = [entry[x_key] for entry in data]
            left = [entry.get(left_key, 0) for entry in data]

            # For right axis, only include points where right_key exists
            right_points = [(entry[x_key], entry.get(right_key))
                            for entry in data if right_key in entry]
            if right_points:
                right_x, right_values = zip(*right_points)
            else:
                right_x, right_values = [], []

            # Clear axis
            ax.clear()
            
            # Plot both metrics on the same axis
            ax.plot(x_values, left, 'b-', label=left_label)
            if right_x:
                ax.plot(right_x, right_values, 'r-', label=right_label)
            
            x_label = "Epoch" if x_key == "epoch" else "Step"
            ax.set_xlabel(x_label)
            ax.set_ylabel(left_label)
            ax.set_title('Training Progress')
            ax.legend(loc='upper left')
            if startPoint is not None:
                ax.set_xlim(left=startPoint)
            fig.tight_layout()

            if os.path.isabs(output_png) or os.path.dirname(output_png):
                output_path = output_png
            else:
                output_path = os.path.join(log_dir, output_png)

            save_figure_safely(fig, output_path)
    finally:
        plt.close(fig)  # Ensure figure is closed even if an error occurs

def plot_scores_by_width(log_file, output_png):
    """Plot caption-adherence score vs epoch with a separate line per scene width.

    Reads the scores-by-epoch JSONL written by ``evaluate_caption_adherence``. Each entry may
    carry a ``width_scores`` dict mapping scene width -> mean caption score for that epoch
    (widths come back as strings once round-tripped through JSON). One line is drawn per width
    so a model's weakness at a particular size stands out. No-ops when the log has no per-width
    data (e.g. a single-width eval set), so it is always safe to call.
    """
    if not os.path.exists(log_file):
        return

    with open(log_file, 'r') as f:
        data = [json.loads(line) for line in f if line.strip()]

    # Keep only entries that carry both an epoch and a non-empty per-width breakdown.
    points = [(entry["epoch"], entry["width_scores"]) for entry in data
              if "epoch" in entry and isinstance(entry.get("width_scores"), dict) and entry["width_scores"]]
    if not points:
        return

    # Union of widths seen across all epochs, sorted numerically (JSON keys are strings).
    widths = sorted({int(w) for _, ws in points for w in ws})

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)
    try:
        for w in widths:
            xs, ys = [], []
            for epoch, ws in points:
                # Tolerate either string (post-JSON) or int keys.
                val = ws.get(str(w), ws.get(w))
                if val is None:
                    continue
                xs.append(epoch)
                ys.append(val)
            if xs:
                ax.plot(xs, ys, marker='o', label=f"width {w}")

        ax.set_xlabel("Epoch")
        ax.set_ylabel("Caption Score")
        ax.set_title("Caption Adherence by Scene Width")
        ax.legend(loc='best')
        fig.tight_layout()
        save_figure_safely(fig, output_png)
    finally:
        plt.close(fig)


def save_figure_safely(fig, output_path):
    """Save figure to a temporary file first, then move it to the final location"""
    output_path = str(Path(output_path))  # Convert to string path
    
    # Create temporary file with .png extension
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
        tmp_path = tmp_file.name
    
    try:
        # Save to temporary file
        fig.savefig(tmp_path)
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        # Try to move the file to final destination
        # If move fails, try to copy and then delete
        try:
            shutil.move(tmp_path, output_path)
        except OSError:
            shutil.copy2(tmp_path, output_path)
            os.unlink(tmp_path)
    except Exception as e:
        # Clean up temporary file if anything goes wrong
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise e

class Plotter:
    def __init__(self, log_file, update_interval=1.0, left_key='loss', right_key='lr', 
                 left_label='Loss', right_label='Learning Rate', output_png='training_progress.png'):
        self.log_dir = os.path.dirname(log_file)
        self.log_file = log_file
        self.update_interval = update_interval
        self.running = True
        self.output_png = output_png
        self.step_png = _step_png_path(output_png)
        self.left_key = left_key
        self.right_key = right_key
        self.left_label = left_label
        self.right_label = right_label
        
        matplotlib.use('Agg')
        
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop_plotting()
        
    def __del__(self):
        self.stop_plotting()

    def update_plot(self):
        general_update_plot(self.log_file, self.left_key, self.right_key,
                            self.left_label, self.right_label, self.output_png,
                            update_interval=self.update_interval, x_key="epoch")

        general_update_plot(self.log_file, self.left_key, self.right_key,
                            self.left_label, self.right_label, self.step_png,
                            update_interval=self.update_interval, x_key="step")
    
    def start_plotting(self):
        print("Starting plotting in background")
        while self.running:
            self.update_plot()
            time.sleep(self.update_interval)
    
    def stop_plotting(self):
        if hasattr(self, 'running'):  # Check if already stopped
            self.running = False
            self.update_plot()
            print("Plotting stopped")

if __name__ == "__main__":
    main()