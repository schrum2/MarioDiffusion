import tkinter as tk
from tkinter import filedialog
import json
import sys
import os
import level_dataset
import torch
from create_ascii_captions import assign_caption, extract_tileset

class TileViewer(tk.Tk):
    def __init__(self, dataset_path=None, tileset_path=None):
        super().__init__()
        self.title("Tile Dataset Viewer")

        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        self.window_size = min(screen_width, screen_height) * 0.75
        self.tile_size = int(self.window_size / 20)
        self.font_size = max(self.tile_size // 3, 6)  # Reduced font size for tighter display

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
        caption, details = assign_caption(sample['scene'], self.id_to_char, self.char_to_id, self.tile_descriptors, describe_locations = False, describe_absence = False, debug = True, return_details = True)
        sample['caption'] = caption
        sample['details'] = details
        print(f"New caption: {caption}")
        print(details)
        self.redraw()

    def toggle_view_mode(self):
        """Toggle between numeric/character grid and image view modes."""
        self.show_images = not getattr(self, 'show_images', False)
        self.redraw()

    def create_widgets(self):
        frame = tk.Frame(self)
        frame.pack(pady=2)  # Reduced padding for tighter vertical spacing

        load_button = tk.Button(frame, text="Load Dataset & Tileset", command=self.load_files)
        load_button.pack()

        # Add a button to load a trained diffusion model
        self.load_model_button = tk.Button(frame, text="Load Model", command=self.load_model)
        self.load_model_button.pack(pady=2)

        checkbox_frame = tk.Frame(self)
        checkbox_frame.pack(pady=2)  # Reduced padding for tighter vertical spacing
        self.checkbox = tk.Checkbutton(checkbox_frame, text="Show numeric IDs", variable=self.show_ids, command=self.redraw)
        self.checkbox.pack(side=tk.LEFT, padx=5)
        regenerate_button = tk.Button(checkbox_frame, text="Regenerate Caption", command=self.regenerate_caption)
        regenerate_button.pack(side=tk.LEFT, padx=5)

        toggle_view_button = tk.Button(checkbox_frame, text="Toggle View Mode", command=self.toggle_view_mode)
        toggle_view_button.pack(side=tk.LEFT, padx=5)

        self.canvas = tk.Canvas(self, bg="white", width=self.window_size, height=self.window_size - 100)  # Further reduced height to minimize empty space
        self.canvas.pack(pady=1)  # Reduced padding for tighter vertical spacing

        # Add Text widget for captions
        self.caption_text = tk.Text(self, height=3, width=int(self.window_size / 8), wrap=tk.WORD)
        self.caption_text.pack(pady=2)
        self.caption_text.tag_configure("center", justify="center")
        self.caption_text.configure(state="disabled")  # Make it read-only

        nav_frame = tk.Frame(self)
        nav_frame.pack(pady=2, side=tk.BOTTOM)  # Moved navigation buttons closer to the canvas
        tk.Button(nav_frame, text="<< Prev", command=self.prev_sample).pack(side=tk.LEFT, padx=5)
        tk.Button(nav_frame, text="Next >>", command=self.next_sample).pack(side=tk.LEFT, padx=5)

        # Add a button to generate from the current scene (initially disabled)
        self.generate_button = tk.Button(nav_frame, text="Generate From Scene", command=self.generate_from_scene, state=tk.DISABLED)
        self.generate_button.pack(side=tk.LEFT, padx=5)

        # Sample info and jump
        info_frame = tk.Frame(self)
        info_frame.pack(pady=2)  # Reduced padding for tighter vertical spacing
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
            _, self.id_to_char, self.char_to_id, self.tile_descriptors = extract_tileset(tileset_path)
            self.current_sample_idx = 0
            self.redraw()
        except Exception as e:
            print(f"Error loading files: {e}")

    def load_model(self):
        """Load a trained diffusion model."""
        model_path = filedialog.askdirectory(title="Select Model Directory")
        if model_path:
            try:
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                from models.text_diffusion_pipeline import TextConditionalDDPMPipeline
                self.pipeline = TextConditionalDDPMPipeline.from_pretrained(model_path).to(self.device)
                print(f"Model loaded from {model_path}")
                self.generate_button.config(state=tk.NORMAL)  # Enable the generate button
            except Exception as e:
                print(f"Error loading model: {e}")

    def generate_from_scene(self):
        """Generate a new level from the current scene using the loaded model."""
        if not hasattr(self, 'pipeline') or not self.pipeline:
            print("No model loaded.")
            return

        if not self.dataset:
            print("No dataset loaded.")
            return

        sample = self.dataset[self.current_sample_idx]
        input_scene = sample['scene']
        input_scene = torch.tensor(input_scene, device=self.device)  # Ensure input_scene is on the same device as the model

        try:
            output = self.pipeline(
                batch_size=1,
                input_scene=input_scene,
                num_inference_steps=50,
                guidance_scale=7.5,
                height=len(input_scene),
                width=len(input_scene[0])
            )
            print("Generated new level from scene.")
            from level_dataset import visualize_samples
            # Convert the output tensor to an image using visualize_samples
            generated_image = visualize_samples(output.images)
            # Ensure the output from visualize_samples is a PIL Image
            if isinstance(generated_image, list):
                generated_image = generated_image[0]  # Handle list case by taking the first element
            generated_image.show()  # Display the generated image
        except Exception as e:
            print(f"Error during generation: {e}")

    def redraw(self):
        if not self.dataset:
            return

        self.canvas.delete("all")
        sample = self.dataset[self.current_sample_idx]

        # Generate unique colors for caption phrases
        import random
        random.seed(42)  # Ensure consistent colors across redraws
        phrase_colors = {}
        if 'details' in sample:
            for phrase in sample['details']:
                phrase_colors[phrase] = f"#{random.randint(0, 255):02x}{random.randint(0, 255):02x}{random.randint(0, 255):02x}"

        print(phrase_colors)

        if getattr(self, 'show_images', False):
            # Display as image using visualize_samples
            from level_dataset import visualize_samples
            import PIL.ImageTk
            from PIL import Image
            one_hot_scene = torch.nn.functional.one_hot(
                torch.tensor(sample['scene'], dtype=torch.long),
                num_classes=15
            ).float().permute(2, 0, 1).unsqueeze(0)  # Add batch dimension

            image = visualize_samples(one_hot_scene)
            if isinstance(image, list):
                image = image[0]  # Handle list case by taking the first element
            photo_image = PIL.ImageTk.PhotoImage(image)
            self.canvas.create_image(
                self.window_size // 2, self.window_size // 2, image=photo_image, anchor="center"
            )
            self.photo_image = photo_image  # Keep a reference to avoid garbage collection
        else:
            # Display as numeric/character grid
            font = ("Courier", self.font_size)
            colors = level_dataset.colors()

            for y in range(16):
                for x in range(16):
                    tile_id = sample['scene'][y][x]
                    text = str(tile_id) if self.show_ids.get() else self.id_to_char.get(tile_id, '?')
                    # Convert (r, g, b) float tuple to hex color string
                    r, g, b = colors[tile_id]
                    color_hex = f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}"

                    # Determine background color based on details
                    bg_color = "white"
                    if 'details' in sample:
                        for phrase, coords in sample['details'].items():
                            if (y, x) in coords:
                                bg_color = phrase_colors[phrase]
                                break

                    # Draw background rectangle
                    self.canvas.create_rectangle(
                        x * self.tile_size, y * self.tile_size,
                        (x + 1) * self.tile_size, (y + 1) * self.tile_size,
                        fill=bg_color, outline=""
                    )

                    # Draw text
                    self.canvas.create_text(
                        x * self.tile_size + self.tile_size // 2,
                        y * self.tile_size + self.tile_size // 2,
                        text=text,
                        font=font,
                        anchor="center",
                        fill=color_hex
                    )

        # Update caption text widget
        self.caption_text.configure(state="normal")
        self.caption_text.delete("1.0", tk.END)
        
        caption_text = sample['caption']
        caption_parts = caption_text.split('.')
        
        for part in caption_parts:
            part = part.strip()
            if part:
                part = part + ". "  # Add back period and space
                color = phrase_colors.get(part.strip('.'), "black")  # Remove period for color lookup
                self.caption_text.tag_configure(color, foreground=color)
                self.caption_text.insert(tk.END, part, (color, "center"))
        
        self.caption_text.configure(state="disabled")

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
        tileset_path = sys.argv[2] if len(sys.argv) == 3 else r'..\TheVGLC\Super Mario Bros\smb.json'
        if not os.path.isfile(dataset_path) or not os.path.isfile(tileset_path):
            print("Invalid file paths provided. Ignoring command-line files.")
            dataset_path = tileset_path = None

    app = TileViewer(dataset_path, tileset_path)
    app.mainloop()
