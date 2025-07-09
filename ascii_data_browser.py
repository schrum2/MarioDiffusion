import tkinter as tk
from tkinter import ttk
from tkinter import filedialog, messagebox  # Add messagebox for feedback
from PIL import Image  # Ensure PIL.Image is imported
import PIL.ImageTk  # Ensure PIL.ImageTk is imported
import json
import sys
import os
import level_dataset
import torch
from create_ascii_captions import assign_caption
from LR_create_ascii_captions import assign_caption as lr_assign_caption
from MM_create_ascii_captions import assign_caption as mm_assign_caption
from captions.util import extract_tileset 
import util.common_settings as common_settings
import random
import colorsys
from util.sampler import scene_to_ascii
from util.sampler import SampleOutput
from models.pipeline_loader import get_pipeline
#from LodeRunner.loderunner.graphics import *


class TileViewer(tk.Tk):
    def __init__(self, dataset_path=None, tileset_path=None):
        super().__init__()
        self.title("Tile Dataset Viewer")
        self.added_sample_indexes = []
        self.composed_thumbnails = []
        self.selected_thumb_idx = None  # Track selected thumbnail

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
        self.current_pil_image = None  # Store the current PIL image for saving
        self.canvas_context_menu = tk.Menu(self, tearoff=0)
        self.canvas_context_menu.add_command(
            label="Save Image As...",
            command=self.save_current_image_as
        )
        self.canvas.bind("<Button-3>", self.show_canvas_context_menu)
        self.canvas.bind("<Control-Button-1>", self.show_canvas_context_menu)  # For macOS

    def show_canvas_context_menu(self, event):
        if getattr(self, 'show_images', False) and self.current_pil_image is not None:
            try:
                self.canvas_context_menu.tk_popup(event.x_root, event.y_root)
            finally:
                self.canvas_context_menu.grab_release()

    def save_current_image_as(self):
        if self.current_pil_image is None:
            messagebox.showerror("Error", "No image to save.")
            return
        default_filename = f"scene_{self.current_sample_idx + 1}.png"
        file_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[
                ("PNG files", "*.png"),
                ("JPEG files", "*.jpg"),
                ("All files", "*.*")
            ],
            title="Save Image As",
            initialfile=default_filename
        )
        if file_path:
            try:
                self.current_pil_image.save(file_path)
                messagebox.showinfo("Success", f"Image saved successfully to:\n{file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save image:\n{str(e)}")

    def regenerate_caption(self):
        print("Regenerating caption...")
        if not self.dataset:
            return
        sample = self.dataset[self.current_sample_idx]
        # Example: check for Lode Runner by a property or filename
        if self.game.get()=="LR":
            caption, details = lr_assign_caption(
                sample['scene'],
                self.id_to_char,
                self.char_to_id,
                self.tile_descriptors,
                describe_locations=False,
                describe_absence=self.describe_absence.get(),
                debug=True,
                return_details=True
            )
        if self.game.get()=="MM-Full" or self.game.get()=="MM-Simple":
            s = sample['caption'] #Done for clarity
            # mm_assign_caption requires an extra argument for some encoded data that the level parser finds. This code moves those keys along
            data = {
                # String parsing to find entrance key
                "entrance_direction": s[s.find("entrance direction")+len("entrance direction ") : s.find(".", s.find("entrance direction"))],
                #String parsing to find exit key
                "exit_direction": s[s.find("exit direction")+len("exit direction ") : s.find(".", s.find("exit direction"))]
            }

            caption, details = mm_assign_caption(
                sample['scene'],
                self.id_to_char,
                self.char_to_id,
                self.tile_descriptors,
                describe_locations=False,
                describe_absence=self.describe_absence.get(),
                data=data,
                debug=True,
                return_details=True
            )
        # If not Lode Runner or Mega Man than Mario
        else:
            caption, details = assign_caption(
                sample['scene'],
                self.id_to_char,
                self.char_to_id,
                self.tile_descriptors,
                describe_locations=False,
                describe_absence=self.describe_absence.get(),
                debug=True,
                return_details=True
            )
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
        # Make it read-only but selectable/copyable
        self.caption_text.bind("<Key>", lambda e: "break")
        self.caption_text.bind("<Button-2>", lambda e: "break")  # Middle click paste
        self.caption_text.bind("<Control-v>", lambda e: "break")
        self.caption_text.bind("<Control-V>", lambda e: "break")
        self.caption_text.bind("<Delete>", lambda e: "break")
        self.caption_text.bind("<BackSpace>", lambda e: "break")
        # Add copy support
        self.caption_text.bind("<Control-c>", self.copy_caption_text)
        self.caption_text.bind("<Control-C>", self.copy_caption_text)
        self.caption_text.bind("<Command-c>", self.copy_caption_text)
        self.caption_text.bind("<Command-C>", self.copy_caption_text)
        # Add right-click context menu for copy
        self.caption_context_menu = tk.Menu(self, tearoff=0)
        self.caption_context_menu.add_command(label="Copy", command=self.copy_caption_text)
        self.caption_text.bind("<Button-3>", self.show_caption_context_menu)
        self.caption_text.bind("<Control-Button-1>", self.show_caption_context_menu)  # For Mac

        # Combined navigation and info frame
        nav_info_frame = tk.Frame(self)
        nav_info_frame.pack(pady=2)  # Place above composed controls and thumbnails

        # Sample info and jump
        self.sample_label = tk.Label(nav_info_frame, text="Sample: 0 / 0")
        self.sample_label.pack(side=tk.LEFT, padx=5)

        tk.Label(nav_info_frame, text="Jump to:").pack(side=tk.LEFT)
        self.jump_entry = tk.Entry(nav_info_frame, width=5)
        self.jump_entry.pack(side=tk.LEFT)
        self.jump_entry.bind("<Return>", self.jump_to_sample)

        # Generate button (initially disabled)
        self.generate_button = tk.Button(nav_info_frame, text="Generate From Scene", command=self.generate_from_scene, state=tk.DISABLED)
        self.generate_button.pack(side=tk.LEFT, padx=20)

        # Steps input field
        tk.Label(nav_info_frame, text="Steps:").pack(side=tk.LEFT)
        self.steps_entry = tk.Entry(nav_info_frame, width=4)
        self.steps_entry.insert(0, "50")  # Default value
        self.steps_entry.config(state=tk.DISABLED)  # Initially disabled
        self.steps_entry.pack(side=tk.LEFT, padx=20)

        # Navigation buttons
        tk.Button(nav_info_frame, text="<< Prev", command=self.prev_sample).pack(side=tk.LEFT, padx=10)
        tk.Button(nav_info_frame, text="Next >>", command=self.next_sample).pack(side=tk.LEFT, padx=10)

        # Composed level controls (below navigation)
        self.composed_frame = tk.Frame(self)
        self.composed_frame.pack(pady=(10, 2))

        # Add buttons to test play the composed level
        self.play_composed_button = tk.Button(self.composed_frame, text="Play Composed Level", command=self.play_composed_level)
        self.play_composed_button.pack(side=tk.LEFT, padx=2)
        self.astar_composed_button = tk.Button(self.composed_frame, text="Use A* on Composed Level", command=self.astar_composed_level)
        self.astar_composed_button.pack(side=tk.LEFT, padx=2)

        # Checkbox for switching between original and SNES graphics
        self.use_snes_graphics = tk.BooleanVar(value=False)
        self.graphics_checkbox = ttk.Checkbutton(
            self.composed_frame,
            text="Use SNES Graphics",
            variable=self.use_snes_graphics
        )
        self.graphics_checkbox.pack(side=tk.LEFT, padx=2)

        # Add button to save composed level
        self.save_composed_button = tk.Button(
            self.composed_frame,
            text="Save Composed Level",
            command=self.save_composed_level
        )
        self.save_composed_button.pack(side=tk.LEFT, padx=2)

        # Add button to add current scene to composed level
        self.add_to_composed_level_button = tk.Button(
            self.composed_frame,
            text="Add To Level",
            command=self.add_to_composed_level
        )
        self.add_to_composed_level_button.pack(side=tk.LEFT, padx=2)

        # Controls for thumbnail manipulation: Move/Delete/Clear all
        tk.Button(self.composed_frame, text="Move Left", command=self.move_selected_thumbnail_left).pack(side=tk.LEFT, padx=2)
        tk.Button(self.composed_frame, text="Move Right", command=self.move_selected_thumbnail_right).pack(side=tk.LEFT, padx=2)
        tk.Button(self.composed_frame, text="Delete", command=self.delete_selected_thumbnail).pack(side=tk.LEFT, padx=2)
        self.clear_composed_button = tk.Button(self.composed_frame, text="Clear Composed Level", command=self.clear_composed_level)
        self.clear_composed_button.pack(side=tk.LEFT, padx=2)
        
        # Thumbnails for composed level
        self.composed_thumb_frame = tk.Frame(self)
        self.composed_thumb_frame.pack(fill=tk.X)

        
        # Game selection
        # Mapping from the game on the display, to the actual internal names of each game
        self.game_display_to_real_mapping = {
            "Mario": "Mario",
            "Lode Runner": "LR",
            "Mega Man (Simple)": "MM-Simple",
            "Mega Man (Full)": "MM-Full"
        }
        
        #Method called every time the dropdown is updated to use the mapping, and putting it in self.game
        def on_game_select(Event=None):
            game_display_var = self.game_display_var.get()
            self.game.set(self.game_display_to_real_mapping.get(game_display_var, game_display_var))
        
        #Creating the game dropdown
        self.game_display_var = tk.StringVar(value="Mario")
        self.game = tk.StringVar(value=self.game_display_to_real_mapping[self.game_display_var.get()])
        self.game_label = ttk.Label(self.composed_frame, text="Select Game:", style="TLabel")
        self.game_label.pack()
        self.game_dropdown = ttk.Combobox(self.composed_frame, textvariable=self.game_display_var, values=["Mario", "Lode Runner", "Mega Man (Simple)", "Mega Man (Full)"], state="readonly")
        self.game_dropdown.pack()
        self.game_dropdown.bind("<<ComboboxSelected>>", on_game_select)


    # method to enter txt file name and save composed level
    def save_composed_level(self):
        scene = self.merge_selected_scenes()
        if scene:
            # Always open in the current working directory or a subfolder
            initial_dir = os.path.join(os.getcwd(), "Composed Levels")
            os.makedirs(initial_dir, exist_ok=True)  # Ensure the folder exists

            file_path = filedialog.asksaveasfilename(
                defaultextension=".txt",
                filetypes=[("Text files", "*.txt")],
                title="Save Composed Level As",
                initialdir=initial_dir
            )
            if file_path:
                # Convert scene to character grid
                char_grid = scene_to_ascii(scene, self.id_to_char)
                # Write to file
                try:
                    with open(file_path, "w") as f:
                        for line in char_grid:
                            f.write(line + "\n")
                    print(f"Composed level saved to {file_path}")
                except Exception as e:
                    print(f"Failed to save composed level: {e}")
            else:
                print("Save operation cancelled.")
        else:
            print("No composed scene to save.")

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
        self.dataset_path = dataset_path
        try:
            with open(dataset_path, 'r') as f:
                self.dataset = json.load(f)

            # Is designed to typically expect both scenes and captions, but if there are only level scenes,
            # convert the data format
            if isinstance(self.dataset, list) and all(isinstance(item, list) for item in self.dataset):
                # Convert to dict format with empty caption
                self.dataset = [{'scene': item, 'caption': ''} for item in self.dataset]

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
                self.pipeline = get_pipeline(model_path).to(self.device)
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
            generated_image = visualize_samples(output.images, game=self.game.get())
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
        from captions.LR_caption_match import TOPIC_KEYWORDS as LR_TOPIC_KEYWORDS
        from captions.MM_caption_match import TOPIC_KEYWORDS as MM_TOPIC_KEYWORDS
        # Generate a palette of distinct colors algorithmically
        # See if running Lode Runner
        if self.game.get()=="LR":
            TOPIC_KEYWORDS = LR_TOPIC_KEYWORDS
        elif self.game.get()=="MM-Simple" or self.game.get()=="MM-Full":
            TOPIC_KEYWORDS = MM_TOPIC_KEYWORDS
        # If not Lode Runner or Mega Man, use the default topic keywords of Mario
        else:
            TOPIC_KEYWORDS = TOPIC_KEYWORDS
        num_topics = len(TOPIC_KEYWORDS)
        topic_colors = {}

        # Golden ratio conjugate for hue stepping
        golden_ratio_conjugate = 0.618033988749895
        h = random.random()  # Start at a random point

        for topic in TOPIC_KEYWORDS:
            # Step through the hue wheel using the golden ratio
            h = (h + golden_ratio_conjugate) % 1
            # Optionally, vary lightness and saturation a bit for more distinction
            saturation = 0.7 + 0.2 * random.random()  # 0.7-0.9
            lightness = 0.45 + 0.1 * random.random()  # 0.45-0.55
            r, g, b = colorsys.hls_to_rgb(h, lightness, saturation)
            topic_colors[topic] = f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}"

        # Map phrases in the sample to their corresponding topic colors
        phrase_colors = {}
        if 'details' in sample:
            for phrase in sample['details']:
                for topic in TOPIC_KEYWORDS:
                    if topic in phrase:
                        phrase_colors[phrase] = topic_colors[topic]
                        break  # Stop at the first matching topic

        

        if getattr(self, 'show_images', False):
            # Display as image using visualize_samples
            from level_dataset import visualize_samples
            import PIL.ImageTk
            from PIL import Image

            #Get the right size for the one-hot encoding
            if self.game.get()=="Mario":
                num_classes = common_settings.MARIO_TILE_COUNT
            elif self.game.get()=="LR":
                num_classes = common_settings.LR_TILE_COUNT
            elif self.game.get()=="MM-Simple":
                num_classes = common_settings.MM_SIMPLE_TILE_COUNT
            else: #Goes to MM-Full if all other cases fail
                num_classes = common_settings.MM_FULL_TILE_COUNT
            
            
            one_hot_scene = torch.nn.functional.one_hot(
                torch.tensor(sample['scene'], dtype=torch.long),
                num_classes=num_classes
            ).float().permute(2, 0, 1).unsqueeze(0)  # Add batch dimension

            image = visualize_samples(one_hot_scene, game=self.game.get())
            if isinstance(image, list):
                image = image[0]  # Handle list case by taking the first element
            # Convert to PIL Image if needed
            if not isinstance(image, Image.Image):
                image = Image.fromarray(image)
            self.current_pil_image = image  # Store for saving
            # --- Resize image to fit canvas ---
            canvas_width = int(self.canvas['width'])
            canvas_height = int(self.canvas['height'])
            img_width, img_height = image.size
            scale = min(canvas_width / img_width, canvas_height / img_height, 1.0)
            display_image = image
            if scale < 1.0:
                new_size = (int(img_width * scale), int(img_height * scale))
                display_image = image.resize(new_size, Image.Resampling.NEAREST)
            photo_image = PIL.ImageTk.PhotoImage(display_image)
            self.canvas.create_image(
                canvas_width // 2, canvas_height // 2, image=photo_image, anchor="center"
            )
            self.photo_image = photo_image  # Keep a reference to avoid garbage collection
        else:
            self.current_pil_image = None  # No image to save in non-image mode
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
        # Do not set state to disabled, so user can select/copy
        # self.caption_text.configure(state="disabled")

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
        self.added_sample_indexes.append(idx)
        # Create a thumbnail for the scene
        from level_dataset import visualize_samples
        scene = self.dataset[idx]['scene']
        one_hot_scene = torch.nn.functional.one_hot(
            torch.tensor(scene, dtype=torch.long),
            num_classes=len(self.id_to_char)
        ).float().permute(2, 0, 1).unsqueeze(0)
        image = visualize_samples(one_hot_scene, game=self.game.get())
        if isinstance(image, list):
            image = image[0]
        # Convert to PIL Image if needed
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        thumb = image.copy()
        thumb.thumbnail((64, 64), Image.Resampling.NEAREST)
        photo = PIL.ImageTk.PhotoImage(thumb)
        self.composed_thumbnails.append(photo)  # Prevent GC
        self.redraw_composed_thumbnails()  # Use new redraw method

    def redraw_composed_thumbnails(self):
        # Clear frame
        for widget in self.composed_thumb_frame.winfo_children():
            widget.destroy()
        # Redraw all thumbnails
        for i, photo in enumerate(self.composed_thumbnails):
            borderwidth = 4 if i == self.selected_thumb_idx else 1
            relief = "solid" if i == self.selected_thumb_idx else "flat"
            label = tk.Label(self.composed_thumb_frame, image=photo, borderwidth=borderwidth, relief=relief)
            label.pack(side=tk.LEFT, padx=2)
            label.bind("<Button-1>", lambda e, idx=i: self.select_thumbnail(idx))

    def select_thumbnail(self, idx):
        self.selected_thumb_idx = idx
        self.redraw_composed_thumbnails()

    def delete_selected_thumbnail(self):
        if self.selected_thumb_idx is not None and 0 <= self.selected_thumb_idx < len(self.added_sample_indexes):
            del self.added_sample_indexes[self.selected_thumb_idx]
            del self.composed_thumbnails[self.selected_thumb_idx]
            # Adjust selection
            if self.selected_thumb_idx >= len(self.composed_thumbnails):
                self.selected_thumb_idx = len(self.composed_thumbnails) - 1
            if self.selected_thumb_idx < 0:
                self.selected_thumb_idx = None
            self.redraw_composed_thumbnails()

    def move_selected_thumbnail_left(self):
        idx = self.selected_thumb_idx
        if idx is not None and idx > 0:
            # Swap with previous
            self.added_sample_indexes[idx - 1], self.added_sample_indexes[idx] = self.added_sample_indexes[idx], self.added_sample_indexes[idx - 1]
            self.composed_thumbnails[idx - 1], self.composed_thumbnails[idx] = self.composed_thumbnails[idx], self.composed_thumbnails[idx - 1]
            self.selected_thumb_idx -= 1
            self.redraw_composed_thumbnails()

    def move_selected_thumbnail_right(self):
        idx = self.selected_thumb_idx
        if idx is not None and idx < len(self.added_sample_indexes) - 1:
            # Swap with next
            self.added_sample_indexes[idx + 1], self.added_sample_indexes[idx] = self.added_sample_indexes[idx], self.added_sample_indexes[idx + 1]
            self.composed_thumbnails[idx + 1], self.composed_thumbnails[idx] = self.composed_thumbnails[idx], self.composed_thumbnails[idx + 1]
            self.selected_thumb_idx += 1
            self.redraw_composed_thumbnails()

    def clear_composed_level(self):
        self.added_sample_indexes.clear()
        self.composed_thumbnails.clear()
        self.selected_thumb_idx = None
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
            level = self.get_sample_output(scene, use_snes_graphics=self.use_snes_graphics.get())
            if self.game.get()=="LR" and not self.validate_lode_runner_level(scene):
                print("Invalid Lode Runner level. Cannot play.")
                return  # Stop playing if level is invalid
            level.play(
                game=self.game.get(),
                level_idx=(self.added_sample_indexes[0] + 1) if self.added_sample_indexes else 1,
                dataset_path=self.dataset_path if hasattr(self, 'dataset_path') else None
            )

    def astar_composed_level(self):
        scene = self.merge_selected_scenes()
        if scene:
            level = self.get_sample_output(scene, use_snes_graphics=self.use_snes_graphics.get())
            console_output = level.run_astar()
            print(console_output)

    def validate_lode_runner_level(self, scene):
        # Check rectangularity
        width = len(scene[0])
        for row in scene:
            if len(row) != width:
                print("Level is not rectangular!")
                return False

        # Check size (e.g., 32x32)
        if len(scene) != 32 or width != 32:
            print(f"Level is not 32x32! Got {len(scene)}x{width}")
            return False

        # Check for player spawn
        player_found = any(self.id_to_char[tile] == 'M' for row in scene for tile in row)
        if not player_found:
            print("No player spawn found!")
            return False

        # Check for at least one gold (if required)
        gold_found = any(self.id_to_char[tile] == 'G' for row in scene for tile in row)
        if not gold_found:
            print("No gold found!")
            return False  # Uncomment if gold is required

        # Check for at least one valid move for the player
        # (You can expand this to check for actual valid moves if needed)

        print("Level validation passed.")
        return True

    def on_close(self):
        self.destroy()
        sys.exit(0)

    def get_sample_output(self, scene, use_snes_graphics=False):
        char_grid = scene_to_ascii(scene, self.id_to_char)
        return SampleOutput(level=char_grid, use_snes_graphics=use_snes_graphics)

    def show_caption_context_menu(self, event):
        try:
            self.caption_context_menu.tk_popup(event.x_root, event.y_root)
        finally:
            self.caption_context_menu.grab_release()

    def copy_caption_text(self, event=None):
        try:
            selection = self.caption_text.get(tk.SEL_FIRST, tk.SEL_LAST)
        except tk.TclError:
            # No selection, copy all
            selection = self.caption_text.get("1.0", tk.END)
        self.clipboard_clear()
        self.clipboard_append(selection)
        return "break"

if __name__ == "__main__":
    # Command-line argument parsing
    dataset_path = None
    tileset_path = None
    if len(sys.argv) == 3 or len(sys.argv) == 2:
        dataset_path = sys.argv[1]
        tileset_path = sys.argv[2] if len(sys.argv) == 3 else common_settings.MARIO_TILESET
        if not os.path.isfile(dataset_path) or not os.path.isfile(tileset_path):
            print("Invalid file paths provided. Ignoring command-line files.")
            dataset_path = tileset_path = None

    # Debugging
    #print("dataset_path", dataset_path)
    #print("tileset_path", tileset_path)
    app = TileViewer(dataset_path, tileset_path)
    app.protocol("WM_DELETE_WINDOW", app.on_close)
    app.mainloop()