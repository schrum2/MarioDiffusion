import tkinter as tk
from tkinter import filedialog
import json
import sys
import os
import level_dataset
from create_ascii_captions import assign_caption, get_tile_descriptors

class TileViewer(tk.Tk):
    def __init__(self, dataset_path=None, tileset_path=None):
        super().__init__()
        self.title("Tile Dataset Viewer")

        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        self.window_size = min(screen_width, screen_height) * 0.75
        self.tile_size = int(self.window_size / 20)
        self.font_size = max(self.tile_size // 2, 8)

        self.dataset = []
        self.id_to_char = {}
        self.current_sample_idx = 0
        self.show_ids = tk.BooleanVar(value=False)

        # UI
        self.create_widgets()
        self.bind_keys()

        # Optional initial load from command-line
        if dataset_path and tileset_path:
            self.load_files_from_paths(dataset_path, tileset_path)

    def regenerate_caption(self):
        print("Regenerating caption...")
        if not self.dataset:
            return
        sample = self.dataset[self.current_sample_idx]
        caption = assign_caption(sample['scene'], self.id_to_char, self.char_to_id, self.tile_descriptors, describe_locations = False, describe_absence = False, debug = True)
        sample['caption'] = caption
        print(f"New caption: {caption}")
        self.redraw()

    def create_widgets(self):
        frame = tk.Frame(self)
        frame.pack(pady=5)

        load_button = tk.Button(frame, text="Load Dataset & Tileset", command=self.load_files)
        load_button.pack()

        checkbox_frame = tk.Frame(self)
        checkbox_frame.pack(pady=5)
        self.checkbox = tk.Checkbutton(checkbox_frame, text="Show numeric IDs", variable=self.show_ids, command=self.redraw)
        self.checkbox.pack(side=tk.LEFT, padx=5)
        regenerate_button = tk.Button(checkbox_frame, text="Regenerate Caption", command=self.regenerate_caption)
        regenerate_button.pack(side=tk.LEFT, padx=5)

        self.canvas = tk.Canvas(self, bg="white", width=self.window_size, height=self.window_size)
        self.canvas.pack()

        nav_frame = tk.Frame(self)
        nav_frame.pack(pady=5)
        tk.Button(nav_frame, text="<< Prev", command=self.prev_sample).pack(side=tk.LEFT, padx=5)
        tk.Button(nav_frame, text="Next >>", command=self.next_sample).pack(side=tk.LEFT, padx=5)

        # Sample info and jump
        info_frame = tk.Frame(self)
        info_frame.pack(pady=5)
        self.sample_label = tk.Label(info_frame, text="Sample: 0 / 0")
        self.sample_label.pack(side=tk.LEFT, padx=5)

        tk.Label(info_frame, text="Jump to:").pack(side=tk.LEFT)
        self.jump_entry = tk.Entry(info_frame, width=5)
        self.jump_entry.pack(side=tk.LEFT)
        self.jump_entry.bind("<Return>", self.jump_to_sample)

    def bind_keys(self):
        self.bind("<Right>", lambda e: self.next_sample())
        self.bind("<Left>", lambda e: self.prev_sample())

    def load_files(self):
        dataset_path = filedialog.askopenfilename(title="Select dataset JSON")
        tileset_path = filedialog.askopenfilename(title="Select tileset JSON")
        if not dataset_path or not tileset_path:
            return
        self.load_files_from_paths(dataset_path, tileset_path)

    def load_files_from_paths(self, dataset_path, tileset_path):
        try:
            with open(dataset_path, 'r') as f:
                self.dataset = json.load(f)
            with open(tileset_path, 'r') as f:
                tileset = json.load(f)
                tile_chars = sorted(tileset['tiles'].keys())
                self.id_to_char = {idx: char for idx, char in enumerate(tile_chars)}
                self.char_to_id = {char: idx for idx, char in enumerate(tile_chars)}
                self.tile_descriptors = get_tile_descriptors(tileset)
                #print(self.id_to_char)
            self.current_sample_idx = 0
            self.redraw()
        except Exception as e:
            print(f"Error loading files: {e}")

    def redraw(self):
        if not self.dataset:
            return

        self.canvas.delete("all")
        sample = self.dataset[self.current_sample_idx]

        font = ("Courier", self.font_size)
        colors = level_dataset.colors()
        for y in range(16):
            for x in range(16):
                tile_id = sample['scene'][y][x]
                text = str(tile_id) if self.show_ids.get() else self.id_to_char.get(tile_id, '?')
                # Convert (r, g, b) float tuple to hex color string
                r, g, b = colors[tile_id]
                color_hex = f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}"
                self.canvas.create_text(
                    x * self.tile_size + self.tile_size // 2,
                    y * self.tile_size + self.tile_size // 2,
                    text=text,
                    font=font,
                    anchor="center",
                    fill=color_hex
                )

        self.canvas.create_text(
            8 * self.tile_size + self.tile_size // 2,
            17 * self.tile_size + self.tile_size // 2,
            text=sample['caption'],
            anchor="center",
            width=self.tile_size * 16
        )

        self.sample_label.config(
            text=f"Sample: {self.current_sample_idx + 1} / {len(self.dataset)}"
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

    def jump_to_sample(self, event=None):
        try:
            idx = int(self.jump_entry.get()) - 1
            if 0 <= idx < len(self.dataset):
                self.current_sample_idx = idx
                self.redraw()
            else:
                print("Index out of range.")
        except ValueError:
            print("Invalid index entered.")

if __name__ == "__main__":
    # Command-line argument parsing
    dataset_path = None
    tileset_path = None
    if len(sys.argv) == 3 or len(sys.argv) == 2:
        dataset_path = sys.argv[1]
        tileset_path = sys.argv[2] if len(sys.argv) == 3 else '..\TheVGLC\Super Mario Bros\smb.json'
        if not os.path.isfile(dataset_path) or not os.path.isfile(tileset_path):
            print("Invalid file paths provided. Ignoring command-line files.")
            dataset_path = tileset_path = None

    app = TileViewer(dataset_path, tileset_path)
    app.mainloop()
