import tkinter as tk
from tkinter import filedialog, ttk
import json

class TileViewer(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Tile Dataset Viewer")
        
        # Get screen size and adjust canvas size dynamically
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        
        self.window_size = min(screen_width, screen_height) * 0.75  # 75% of smaller dimension
        self.tile_size = int(self.window_size / 20)  # Keep padding/margins in mind
        self.font_size = max(self.tile_size // 2, 8)  # Ensure font is readable
        
        self.dataset = []
        self.id_to_char = {}
        self.current_sample_idx = 0
        self.show_ids = tk.BooleanVar(value=False)

        # UI
        self.create_widgets()

    def create_widgets(self):
        frame = tk.Frame(self)
        frame.pack(pady=5)

        load_button = tk.Button(frame, text="Load Dataset & Tileset", command=self.load_files)
        load_button.pack()

        self.checkbox = tk.Checkbutton(self, text="Show numeric IDs", variable=self.show_ids, command=self.redraw)
        self.checkbox.pack()

        # Create dynamically sized canvas
        self.canvas = tk.Canvas(self, bg="white", width=self.window_size, height=self.window_size)
        self.canvas.pack()

        nav_frame = tk.Frame(self)
        nav_frame.pack(pady=5)
        tk.Button(nav_frame, text="<< Prev", command=self.prev_sample).pack(side=tk.LEFT, padx=5)
        tk.Button(nav_frame, text="Next >>", command=self.next_sample).pack(side=tk.LEFT, padx=5)

    def load_files(self):
        dataset_path = filedialog.askopenfilename(title="Select dataset JSON")
        tileset_path = filedialog.askopenfilename(title="Select tileset JSON")
        if not dataset_path or not tileset_path:
            return

        # Load dataset
        with open(dataset_path, 'r') as f:
            self.dataset = json.load(f)

        # Load tileset
        with open(tileset_path, 'r') as f:
            tileset = json.load(f)
            tile_chars = sorted(tileset['tiles'].keys())
            self.id_to_char = {idx: char for idx, char in enumerate(tile_chars)}

        self.current_sample_idx = 0
        self.redraw()

    def redraw(self):
        if not self.dataset:
            return

        self.canvas.delete("all")
        sample = self.dataset[self.current_sample_idx]

        font = ("Courier", self.font_size)

        for y in range(16):
            for x in range(16):
                tile_id = sample[y][x]
                text = str(tile_id) if self.show_ids.get() else self.id_to_char.get(tile_id, '?')
                self.canvas.create_text(
                    x * self.tile_size + self.tile_size // 2,
                    y * self.tile_size + self.tile_size // 2,
                    text=text,
                    font=font,
                    anchor="center"
                )

        self.title(f"Tile Dataset Viewer - Sample {self.current_sample_idx + 1} / {len(self.dataset)}")

    def prev_sample(self):
        if self.current_sample_idx > 0:
            self.current_sample_idx -= 1
            self.redraw()

    def next_sample(self):
        if self.current_sample_idx < len(self.dataset) - 1:
            self.current_sample_idx += 1
            self.redraw()

if __name__ == "__main__":
    app = TileViewer()
    app.mainloop()
