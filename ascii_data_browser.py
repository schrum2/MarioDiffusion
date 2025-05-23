import tkinter as tk
from tkinter import filedialog
import json
import sys
import os
import level_dataset
import torch
from create_ascii_captions import assign_caption
from captions.util import extract_tileset 
import util.common_settings as common_settings

class TileViewer(tk.Tk):
    def __init__(self, dataset_path=None, tileset_path=None):
        super().__init__()
        self.title("Tile Dataset Viewer")

        # Set a more reasonable default window size for typical grids
        self.window_size = 512  # px, fits 16x16 well but still scales for larger grids
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        self.window_size = min(screen_width, screen_height) * 0.65
        self.tile_size = int(self.window_size / 20)
        self.font_size = max(self.tile_size // 4, 6)  # Reduced font size for tighter display

        self.dataset = []
        self.id_to_char = {}
        self.current_sample_idx = 0
        self.show_ids = tk.BooleanVar(value=False)
        #self.describe_locations = tk.BooleanVar(value=False)
        self.describe_absence = tk.BooleanVar(value=False)

        # UI
        self.create_widgets()
        self.bind_keys()

        # Optional initial load from command-line
        if dataset_path and tileset_path:
            self.load_files_from_paths(dataset_path, tileset_path)

        # Lists to added level segments to the composed level
        self.added_sample_indexes = []
        self.composed_thumbnails = []

    def regenerate_caption(self):
        print("Regenerating caption...")
        if not self.dataset:
            return
        sample = self.dataset[self.current_sample_idx]
        caption, details = assign_caption(sample['scene'], 
                                       self.id_to_char, 
                                       self.char_to_id, 
                                       self.tile_descriptors, 
                                       describe_locations=False, #self.describe_locations.get(), 
                                       describe_absence=self.describe_absence.get(), 
                                       debug=True, 
                                       return_details=True)
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
        
        # Create a sub-frame for the caption options
        caption_options_frame = tk.Frame(checkbox_frame)
        caption_options_frame.pack(side=tk.LEFT, padx=5)
        
        # Add checkboxes for caption generation options
        tk.Checkbutton(caption_options_frame, text="Show numeric IDs", variable=self.show_ids, command=self.redraw).pack(anchor=tk.W)
        #tk.Checkbutton(caption_options_frame, text="Describe Locations", variable=self.describe_locations, state=tk.DISABLED).pack(anchor=tk.W)
        tk.Checkbutton(caption_options_frame, text="Describe Absence", variable=self.describe_absence).pack(anchor=tk.W)
        
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

        # Add steps input field
        tk.Label(nav_frame, text="Steps:").pack(side=tk.LEFT)
        self.steps_entry = tk.Entry(nav_frame, width=4)
        self.steps_entry.insert(0, "50")  # Default value
        self.steps_entry.config(state=tk.DISABLED)  # Initially disabled
        self.steps_entry.pack(side=tk.LEFT, padx=2)

        # Sample info and jump
        info_frame = tk.Frame(self)
        info_frame.pack(pady=2)  # Reduced padding for tighter vertical spacing
        self.sample_label = tk.Label(info_frame, text="Sample: 0 / 0")
        self.sample_label.pack(side=tk.LEFT, padx=5)

        tk.Label(info_frame, text="Jump to:").pack(side=tk.LEFT)
        self.jump_entry = tk.Entry(info_frame, width=5)
        self.jump_entry.pack(side=tk.LEFT)
        self.jump_entry.bind("<Return>", self.jump_to_sample)

        # Composed level controls (below navigation)
        self.composed_frame = tk.Frame(self)
        self.composed_frame.pack(pady=(10, 2))

        # Add buttons to add scenes to composed level and test play the level
        self.play_composed_button = tk.Button(self.composed_frame, text="Play Composed Level", command=self.play_composed_level)
        self.play_composed_button.pack(side=tk.LEFT, padx=2)
        self.astar_composed_button = tk.Button(self.composed_frame, text="Use A* on Composed Level", command=self.astar_composed_level)
        self.astar_composed_button.pack(side=tk.LEFT, padx=2)
        self.clear_composed_button = tk.Button(self.composed_frame, text="Clear Composed Level", command=self.clear_composed_level)
        self.clear_composed_button.pack(side=tk.LEFT, padx=2)
        self.add_to_composed_level_button = tk.Button(
            self.composed_frame,
            text="Add To Level",
            command=self.add_to_composed_level
        )
        self.add_to_composed_level_button.pack(side=tk.LEFT, padx=2)
        
        # Thumbnails for composed level
        self.composed_thumb_frame = tk.Frame(self)
        self.composed_thumb_frame.pack(fill=tk.X)

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
            raise e

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
                self.steps_entry.config(state=tk.NORMAL)  # Enable the steps entry
            except Exception as e:
                print(f"Error loading model: {e}")
                self.generate_button.config(state=tk.DISABLED)
                self.steps_entry.config(state=tk.DISABLED)

    def generate_from_scene(self):
        """Generate a new level from the current scene using the loaded model."""
        if not hasattr(self, 'pipeline') or not self.pipeline:
            print("No model loaded.")
            return

        if not self.dataset:
            print("No dataset loaded.")
            return

        # Get number of steps from entry, with validation
        try:
            num_steps = int(self.steps_entry.get())
            if num_steps <= 0:
                raise ValueError("Steps must be positive")
        except ValueError as e:
            print(f"Invalid step count: {e}")
            self.steps_entry.delete(0, tk.END)
            self.steps_entry.insert(0, "50")  # Reset to default
            num_steps = common_settings.NUM_INFERENCE_STEPS

        sample = self.dataset[self.current_sample_idx]
        input_scene = sample['scene']
        input_scene = torch.tensor(input_scene, device=self.device)

        try:
            output = self.pipeline(
                batch_size=1,
                input_scene=input_scene,
                num_inference_steps=num_steps,  # Use the value from entry
                guidance_scale=common_settings.GUIDANCE_SCALE,
                height=len(input_scene),
                width=len(input_scene[0])
            )
            print(f"Generated new level from scene using {num_steps} steps.")
            from level_dataset import visualize_samples
            generated_image = visualize_samples(output.images)
            if isinstance(generated_image, list):
                generated_image = generated_image[0]
            generated_image.show()
        except Exception as e:
            print(f"Error during generation: {e}")

    def create_triangle_coords(self, x, y, num_colors):
        """Create coordinates for triangle partitions based on number of colors"""
        x1, y1 = x * self.tile_size, y * self.tile_size
        x2, y2 = (x + 1) * self.tile_size, (y + 1) * self.tile_size
        xm, ym = (x1 + x2) / 2, (y1 + y2) / 2  # midpoint

        if num_colors == 2:
            # Two right triangles divided by diagonal
            return [
                [(x1, y1), (x2, y1), (x2, y2)],  # upper right triangle
                [(x1, y1), (x1, y2), (x2, y2)]   # lower left triangle
            ]
        elif num_colors == 3:
            # One right triangle, other two split remaining triangle
            return [
                [(x1, y1), (x2, y1), (x2, y2)],          # upper right triangle
                [(x1, y1), (x1, y2), (xm, ym)],          # left triangle
                [(x1, y2), (x2, y2), (xm, ym)]           # bottom triangle
            ]
        elif num_colors == 4:
            # Four triangles meeting at center
            return [
                [(x1, y1), (xm, ym), (x2, y1)],  # top triangle
                [(x2, y1), (xm, ym), (x2, y2)],  # right triangle
                [(x2, y2), (xm, ym), (x1, y2)],  # bottom triangle
                [(x1, y2), (xm, ym), (x1, y1)]   # left triangle
            ]
        else:
            return [[(x1, y1), (x2, y1), (x2, y2), (x1, y2)]]  # full square

    
    def update_tile_and_canvas_size(self, scene):
        """Update tile_size and canvas size so the level fits perfectly inside the window."""
        HEIGHT = len(scene)
        WIDTH = len(scene[0])
        # Compute the largest tile size that fits both dimensions
        tile_size_h = int(self.window_size // HEIGHT)
        tile_size_w = int(self.window_size // WIDTH)
        self.tile_size = min(tile_size_h, tile_size_w)
        # Update canvas size to fit the grid exactly
        canvas_width = self.tile_size * WIDTH
        canvas_height = self.tile_size * HEIGHT
        self.canvas.config(width=canvas_width, height=canvas_height)
        # Make font smaller relative to tile size for better fit
        self.font_size = max(self.tile_size // 3, 6)

    def redraw(self):
        if not self.dataset:
            return

        self.canvas.delete("all")
        sample = self.dataset[self.current_sample_idx]

        if isinstance(sample, list):
            sample = {"scene": sample, "caption": "No caption available."}

        # Dynamically update tile and canvas size for this scene
        self.update_tile_and_canvas_size(sample['scene'])

        # Generate unique colors for caption phrases based on TOPIC_KEYWORDS
        from captions.caption_match import TOPIC_KEYWORDS
        import colorsys
        # Generate a palette of distinct colors algorithmically
        num_topics = len(TOPIC_KEYWORDS)
        topic_colors = {}
        for i, topic in enumerate(TOPIC_KEYWORDS):
            hue = i / num_topics  # Distribute hues evenly across the color wheel
            saturation = 0.7  # Keep saturation high for vivid colors
            lightness = 0.5  # Keep lightness moderate for good visibility
            r, g, b = colorsys.hls_to_rgb(hue, lightness, saturation)
            topic_colors[topic] = f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}"

        # Map phrases in the sample to their corresponding topic colors
        phrase_colors = {}
        if 'details' in sample:
            for phrase in sample['details']:
                for topic in TOPIC_KEYWORDS:
                    if topic in phrase:
                        phrase_colors[phrase] = topic_colors[topic]
                        break  # Stop at the first matching topic

        #print("phrase_colors", phrase_colors)

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

            HEIGHT = len(sample['scene'])
            WIDTH = len(sample['scene'][0])
            for y in range(HEIGHT):
                for x in range(WIDTH):
                    tile_id = sample['scene'][y][x]
                    text = str(tile_id) if self.show_ids.get() else self.id_to_char.get(tile_id, '?')
                    # Convert (r, g, b) float tuple to hex color string
                    r, g, b = colors[tile_id]
                    color_hex = f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}"

                    # Find all matching phrases for this coordinate
                    matching_phrases = []
                    if 'details' in sample:
                        for phrase, coords in sample['details'].items():
                            if (y, x) in coords:
                                matching_phrases.append(phrase)

                    # Draw background based on number of matching phrases
                    if not matching_phrases:
                        # Draw simple white rectangle for no matches
                        self.canvas.create_rectangle(
                            x * self.tile_size, y * self.tile_size,
                            (x + 1) * self.tile_size, (y + 1) * self.tile_size,
                            fill="white", outline=""
                        )
                    else:
                        # Get triangle coordinates based on number of colors
                        triangles = self.create_triangle_coords(x, y, len(matching_phrases))
                        # Draw each triangle with its corresponding phrase color
                        for i, phrase in enumerate(matching_phrases[:4]):  # Limit to 4 colors max
                            coords = []
                            for point in triangles[i]:
                                coords.extend(point)
                            self.canvas.create_polygon(
                                *coords,
                                fill=phrase_colors[phrase],
                                outline=""
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
                part = part + "."  # Add back period
                color = phrase_colors.get(part, "black")  # Remove period for color lookup
                part = part + " " # Add space for readability
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

    def add_to_composed_level(self):
        idx = self.current_sample_idx
        #print("HELLO", idx, self.added_sample_indexes)
        if idx in self.added_sample_indexes:
            return
        #print("WHY?")
        self.added_sample_indexes.append(idx)
        # Create a thumbnail for the scene
        from level_dataset import visualize_samples
        import PIL.ImageTk
        scene = self.dataset[idx]['scene']
        one_hot_scene = torch.nn.functional.one_hot(
            torch.tensor(scene, dtype=torch.long),
            num_classes=len(self.id_to_char)
        ).float().permute(2, 0, 1).unsqueeze(0)
        image = visualize_samples(one_hot_scene)
        if isinstance(image, list):
            image = image[0]
        thumb = image.copy()
        thumb.thumbnail((64, 64))
        photo = PIL.ImageTk.PhotoImage(thumb)
        self.composed_thumbnails.append(photo)  # Prevent GC
        label = tk.Label(self.composed_thumb_frame, image=photo)
        label.pack(side=tk.LEFT, padx=2)

    def clear_composed_level(self):
        self.added_sample_indexes.clear()
        self.composed_thumbnails.clear()
        for widget in self.composed_thumb_frame.winfo_children():
            widget.destroy()

    def merge_selected_scenes(self):
        scenes = [self.dataset[i]['scene'] for i in self.added_sample_indexes]
        if not scenes:
            return None
        num_rows = len(scenes[0])
        if not all(len(scene) == num_rows for scene in scenes):
            raise ValueError("All scenes must have the same number of rows.")
        concatenated_scene = []
        for row_index in range(num_rows):
            new_row = []
            for scene in scenes:
                new_row.extend(scene[row_index])
            concatenated_scene.append(new_row)
        return concatenated_scene

    def play_composed_level(self):
        scene = self.merge_selected_scenes()
        if scene:
            from util.sampler import SampleOutput
            char_grid = []
            for row in scene:
                char_row = "".join([self.id_to_char[num] for num in row])
                char_grid.append(char_row)
            level = SampleOutput(level=char_grid)
            level.play()

    def astar_composed_level(self):
        scene = self.merge_selected_scenes()
        if scene:
            from util.sampler import SampleOutput
            char_grid = []
            for row in scene:
                char_row = "".join([self.id_to_char[num] for num in row])
                char_grid.append(char_row)
            level = SampleOutput(level=char_grid)
            level.run_astar()

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

    # Debugging
    #print("dataset_path", dataset_path)
    #print("tileset_path", tileset_path)
    app = TileViewer(dataset_path, tileset_path)
    app.mainloop()