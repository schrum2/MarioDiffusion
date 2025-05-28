import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
import json
import torch
import gc
from PIL import ImageTk
import sys
from util.gui_shared import ParentBuilder
from models.text_diffusion_pipeline import TextConditionalDDPMPipeline
from level_dataset import visualize_samples, convert_to_level_format
from util.sampler import SampleOutput
from captions.caption_match import compare_captions
from create_ascii_captions import assign_caption
from LR_create_ascii_captions import assign_caption as lr_assign_caption
from captions.util import extract_tileset
import util.common_settings as common_settings

# Add the parent directory to sys.path so sibling folders can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from LodeRunner import main as lr_main
except ImportError:
    lr_main = None  # Handle gracefully if not present

class CaptionBuilder(ParentBuilder):
    def __init__(self, master):
        super().__init__(master) 
                
        # Holds tensors of levels currently on display
        self.current_levels = []
        self.added_image_indexes = []
        self.bottom_thumbnails = []
        self.generated_images = []
        self.generated_scenes = []

        # Frame for caption display
        self.caption_frame = ttk.Frame(master, width=200, borderwidth=2, relief="solid")  # Add border
        self.caption_frame.pack(side=tk.LEFT, fill=tk.Y, expand=False)  # Only fill vertically, don't expand horizontally
        
        self.caption_label = ttk.Label(self.caption_frame, text="Constructed Caption:", font=("Arial", 12, "bold"))
        self.caption_label.pack(pady=5)
        
        self.caption_text = tk.Text(self.caption_frame, height=8, wrap=tk.WORD, state=tk.DISABLED)
        self.caption_text.pack() 
                
        self.negative_prompt_label = ttk.Label(self.caption_frame, text="Negative Prompt:")
        self.negative_prompt_label.pack()
        self.negative_prompt_entry = ttk.Entry(self.caption_frame, width=100)
        self.negative_prompt_entry.pack()
        self.negative_prompt_entry.insert(0, "")
        
        self.num_images_label = ttk.Label(self.caption_frame, text="Number of Images:")
        self.num_images_label.pack()        
        self.num_images_entry = ttk.Entry(self.caption_frame)
        self.num_images_entry.pack()
        self.num_images_entry.insert(0, "4")

        self.seed_label = ttk.Label(self.caption_frame, text="Random Seed:")
        self.seed_label.pack()        
        self.seed_entry = ttk.Entry(self.caption_frame)
        self.seed_entry.pack()
        self.seed_entry.insert(0, "1")

        self.num_steps_label = ttk.Label(self.caption_frame, text="Num Inference Steps:")
        self.num_steps_label.pack()
        self.num_steps_entry = ttk.Entry(self.caption_frame)
        self.num_steps_entry.pack()
        self.num_steps_entry.insert(0, f"{common_settings.NUM_INFERENCE_STEPS}")
        
        self.guidance_label = ttk.Label(self.caption_frame, text="Guidance Scale:")
        self.guidance_label.pack()
        self.guidance_entry = ttk.Entry(self.caption_frame)
        self.guidance_entry.pack()
        self.guidance_entry.insert(0, f"{common_settings.GUIDANCE_SCALE}")

        self.width_label = ttk.Label(self.caption_frame, text="Width (in tiles):")
        self.width_label.pack()
        self.width_entry = ttk.Entry(self.caption_frame)
        self.width_entry.pack()
        self.width_entry.insert(0, f"{common_settings.MARIO_WIDTH}")
                
        self.generate_button = ttk.Button(self.caption_frame, text="Generate Image", command=self.generate_image)
        self.generate_button.pack(pady=5)
                
        self.model_button = ttk.Button(self.checkbox_frame, text="Load Model", command=self.load_model)
        self.model_button.pack(anchor=tk.E)

        # Frame for image display
        self.image_frame = ttk.Frame(master, borderwidth=2, relief="solid")  # Add border
        self.image_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        self.image_canvas = tk.Canvas(self.image_frame, borderwidth=0, highlightthickness=0)
        self.image_scrollbar = ttk.Scrollbar(self.image_frame, orient=tk.VERTICAL, command=self.image_canvas.yview)
        self.image_inner_frame = ttk.Frame(self.image_canvas, borderwidth=2, relief="solid")  # Add border
        self.image_inner_frame.grid_columnconfigure(0, weight=1)  # Allow centering
        
        def resize_inner_frame(event):
            canvas_width = event.width
            self.image_canvas.itemconfig(self.inner_frame_window, width=canvas_width)
        self.inner_frame_window = self.image_canvas.create_window((0, 0), window=self.image_inner_frame, anchor="n", width=self.image_canvas.winfo_width())
        self.image_canvas.bind('<Configure>', resize_inner_frame)
        self.image_inner_frame.bind("<Configure>", lambda e: self.image_canvas.configure(scrollregion=self.image_canvas.bbox("all")))
        self.image_canvas.configure(yscrollcommand=self.image_scrollbar.set)
        
        self.image_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.image_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Bind mousewheel scrolling to the scrollbar for image_inner_frame
        self.image_canvas.bind_all("<MouseWheel>", lambda event: self.image_canvas.yview_scroll(-1 * (event.delta // 120), "units"))
        
        self.checkbox_vars = {}

        self.loaded_model_label = ttk.Label(self.caption_frame, text=f"Using model: Not loaded yet")
        self.loaded_model_label.pack()

        # Frame for composed level controls
        self.composed_frame = ttk.Frame(self.caption_frame)
        self.composed_frame.pack(fill=tk.X, pady=(20, 5))  # 20 pixels above, 5 below

        self.play_composed_button = ttk.Button(self.composed_frame, text="Play Composed Level", command=self.play_composed_level)
        self.play_composed_button.pack(side=tk.LEFT, padx=2)
        self.save_composed_button = ttk.Button(self.composed_frame, text="Save Composed Level", command=self.save_composed_level)
        self.save_composed_button.pack(side=tk.LEFT, padx=2)
        self.clear_composed_button = ttk.Button(self.composed_frame, text="Clear Composed Level", command=self.clear_composed_level)
        self.clear_composed_button.pack(side=tk.LEFT, padx=2)
        self.astar_composed_button = ttk.Button(
            self.composed_frame, 
            text="Use A* on Composed Level", 
            command=self.astar_composed_level
        )
        self.astar_composed_button.pack(side=tk.LEFT, padx=2)

        # Frame for thumbnails with horizontal scrolling
        self.bottom_canvas = tk.Canvas(self.caption_frame, height=70, borderwidth=0, highlightthickness=0)
        self.bottom_scrollbar = ttk.Scrollbar(self.caption_frame, orient=tk.HORIZONTAL, command=self.bottom_canvas.xview)
        self.bottom_frame = ttk.Frame(self.bottom_canvas)

        self.bottom_frame.bind(
            "<Configure>",
            lambda e: self.bottom_canvas.configure(
                scrollregion=self.bottom_canvas.bbox("all")
            )
        )
        self.bottom_canvas.create_window((0, 0), window=self.bottom_frame, anchor="nw")
        self.bottom_canvas.configure(xscrollcommand=self.bottom_scrollbar.set)

        self.bottom_canvas.pack(fill=tk.X, pady=(0, 0))
        self.bottom_scrollbar.pack(fill=tk.X, pady=(0, 10))

        # Game selection
        self.game_var = tk.StringVar(value="Mario")
        self.game_label = ttk.Label(self.caption_frame, text="Select Game:")
        self.game_label.pack()
        self.game_dropdown = ttk.Combobox(self.caption_frame, textvariable=self.game_var, values=["Mario", "Lode Runner"], state="readonly")
        self.game_dropdown.pack()

    def get_patterns(self):
        # Different for LoRA and tile diffusion
        patterns = ["floor", "ceiling", 
                    "pipe", "coin", "platform", "tower", #"wall",
                    "cannon", "staircase", "rectangular", "irregular",
                    "question block", "loose block", "enem"]
        return patterns

    def load_data(self, filepath = None):
        if filepath == None:
            filepath = filedialog.askopenfilename(title="Select JSON File", filetypes=[("JSON", "*.json")])
        if filepath:
            _, self.id_to_char, self.char_to_id, self.tile_descriptors = extract_tileset(tileset_path)

            try:
                phrases_set = set()
                with open(filepath, 'r') as f:
                    dataset = json.load(f)
                    for item in dataset:
                        phrases = item['caption'].split('.')
                        phrases_set.update(phrase.strip() for phrase in phrases if phrase.strip())
                
                self.all_phrases = sorted(list(phrases_set))
                self.create_checkboxes()

                return True
            except FileNotFoundError as e:
                print(f"Error loading data: {e}")
                messagebox.showerror("Error", f"Error loading data: {e}")

        return False
        
    def load_model(self, model = None):
        if model == None:
            model = filedialog.askopenfilename(title="Select Model Index", filetypes=[("JSON", "*.json")])
            if model: # removed model model_index.json
                model = os.path.dirname(model)
        if model:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.pipe = TextConditionalDDPMPipeline.from_pretrained(model).to(self.device)

            filename = os.path.splitext(os.path.basename(model))[0]
            self.loaded_model_label["text"] = f"Using model: {filename}"
    
            # Enable or disable negative prompt entry based on pipeline support
            if hasattr(self.pipe, "supports_negative_prompt") and self.pipe.supports_negative_prompt:
                self.negative_prompt_entry.config(state=tk.NORMAL)
            else:
                self.negative_prompt_entry.delete(0, tk.END)
                self.negative_prompt_entry.config(state=tk.DISABLED)

    def update_caption(self):
        self.selected_phrases = [phrase for phrase, var in self.checkbox_vars.items() if var.get()]
        new_caption = ". ".join(self.selected_phrases) + "." if self.selected_phrases else ""
        
        self.caption_text.config(state=tk.NORMAL)
        self.caption_text.delete(1.0, tk.END)
        self.caption_text.insert(tk.END, new_caption)
        self.caption_text.config(state=tk.DISABLED)
    
    def generate_image(self):
        # clear the previous images
        self.generated_images = []
        self.generated_scenes = []

        print("Generating")
        prompt = self.caption_text.get("1.0", tk.END).strip()
        negative_prompt = self.negative_prompt_entry.get().strip()
        num_images = int(self.num_images_entry.get())        
        param_values = {
            "num_inference_steps": int(self.num_steps_entry.get()),
            "guidance_scale": float(self.guidance_entry.get()),
            "width": int(self.width_entry.get()),
            "output_type": "tensor"
        }

        # Include caption if desired
        if prompt != "":
            param_values["caption"] = prompt
        # Include negative prompt if provided
        if negative_prompt != "":
            param_values["negative_prompt"] = negative_prompt

        generator = torch.Generator(self.device).manual_seed(int(self.seed_entry.get()))
        
        self.image_inner_frame
        for widget in self.image_inner_frame.winfo_children():
            widget.destroy()

        self.current_levels = []

        # Debugging print statements to trace the issue
        print("Starting image generation...")
        self.image_inner_frame.update_idletasks()  # Force an update to ensure the frame is fully rendered
        frame_width = self.image_inner_frame.winfo_width()
        print(f"Frame width after update_idletasks: {frame_width}")

        # Use a cached frame width if available and valid
        if hasattr(self, 'cached_frame_width') and self.cached_frame_width > 1:
            frame_width = self.cached_frame_width
            print(f"Using cached frame width: {frame_width}")
        elif frame_width <= 1:  # If the width is invalid or too small
            frame_width = self.image_canvas.winfo_width() // 2  # Use third of the parent canvas width as a fallback
            print(f"Frame width was invalid, using third of canvas width: {frame_width}")
        else:
            # Cache the valid frame width for future use
            self.cached_frame_width = frame_width
            print(f"Caching frame width: {frame_width}")

        for i in range(num_images):
            print(f"Generating image {i + 1} of {num_images}...")
            images = self.pipe(generator=generator, **param_values).images
            self.current_levels.append(images[0].cpu().detach().numpy())

            sample_tensor = images[0].unsqueeze(0)
            sample_indices = convert_to_level_format(sample_tensor)
            scene = sample_indices[0].tolist()
            self.generated_scenes.append(scene)
            actual_caption = assign_caption(scene, self.id_to_char, self.char_to_id, self.tile_descriptors, False, False)

            pil_img = visualize_samples(images)
            self.generated_images.append(pil_img)
            img_tk = ImageTk.PhotoImage(pil_img)

            compare_score, exact_matches, partial_matches, excess_phrases = compare_captions(prompt, actual_caption, return_matches=True)

            img_frame = ttk.Frame(self.image_inner_frame)
            img_frame.grid(row=i, column=0, pady=10, sticky="n")  # Center each image frame horizontally


            print(f"Image {i + 1} dimensions: width={img_tk.width()}, height={img_tk.height()}")

            # Check if the image width exceeds the frame width and scale it down if necessary
            if img_tk.width() > frame_width:
                scale_factor = frame_width / img_tk.width()
                new_width = frame_width
                new_height = int(img_tk.height() * scale_factor)
                img_tk = img_tk._PhotoImage__photo.subsample(img_tk.width() // new_width, img_tk.height() // new_height)
                print(f"Image {i + 1} scaled to: width={new_width}, height={new_height}")

            label = ttk.Label(img_frame, image=img_tk)
            label.image = img_tk
            label.pack()

            # Create a Text widget to allow colored text
            caption_text = tk.Text(img_frame, wrap=tk.WORD, width=40, height=5, state=tk.DISABLED)
            caption_text.pack(pady=(5, 10))

            # Enable editing temporarily to insert text
            caption_text.config(state=tk.NORMAL)

            # Define tags for different colors
            caption_text.tag_configure("green", foreground="green")
            caption_text.tag_configure("yellow", foreground="#CCCC00")  # Darker yellow
            caption_text.tag_configure("red", foreground="red")

            # Insert text with tags
            for phrase in exact_matches:
                caption_text.insert(tk.END, phrase + ". ", "green")
            for phrase in partial_matches:
                caption_text.insert(tk.END, phrase + ". ", "yellow")
            for phrase in excess_phrases:
                caption_text.insert(tk.END, phrase + ". ", "red")

            # Disable editing again
            caption_text.config(state=tk.DISABLED)

            # And score
            #score_label = ttk.Label(img_frame, text=f"Comparison Score: {compare_score}", wraplength=300)
            #score_label.pack(pady=(5, 10))  # Add padding: 5px top, 10px bottom

            # Check if the scene is wider than standard number of tiles and process segments if necessary
            avg_segment_score = None
            if len(scene[0]) > common_settings.MARIO_WIDTH:
                from captions.caption_match import process_scene_segments
                avg_segment_score, _, _ = process_scene_segments(
                    scene=scene,
                    segment_width=common_settings.MARIO_WIDTH,
                    prompt=prompt,
                    id_to_char=self.id_to_char,
                    char_to_id=self.char_to_id,
                    tile_descriptors=self.tile_descriptors,
                    describe_locations=False,
                    describe_absence=False
                )

            # Update the score label text
            if avg_segment_score is not None:
                score_label_text = f"""Comparison Score: {compare_score}
Average Segment Score: {avg_segment_score}"""
            else:
                score_label_text = f"Comparison Score: {compare_score}"

            score_label = ttk.Label(img_frame, text=score_label_text, wraplength=300)
            score_label.pack(pady=(5, 10))  # Add padding: 5px top, 10px bottom
    
            # Create a frame for buttons
            button_frame = ttk.Frame(img_frame)
            button_frame.pack(pady=5)
    
            # Add Play button
            play_button = ttk.Button(
                button_frame, 
                text="Play", 
                command=lambda idx=i: self.play_level(idx)
            )
            play_button.pack(side=tk.LEFT, padx=5)
    
            # Add Use A* button
            astar_button = ttk.Button(
                button_frame, 
                text="Use A*", 
                command=lambda idx=i: self.use_astar(idx)
            )
            astar_button.pack(side=tk.LEFT, padx=5)

            # Add "Add To Level" button
            add_button = ttk.Button(
                button_frame,
                text="Add To Level",
                command=lambda idx=i: self.add_to_composed_level(idx)
            )
            add_button.pack(side=tk.LEFT, padx=5)

            del images, sample_tensor, sample_indices, scene  # Delete unused tensors
            torch.cuda.empty_cache()  # Clear the cache
            gc.collect()  # Force garbage collection

        print("Image generation completed.")
        #print(self.current_levels)

    def add_to_composed_level(self, idx):
        self.added_image_indexes.append(idx)
        img = self.generated_images[idx].copy()
        img.thumbnail((64, 64))
        photo = ImageTk.PhotoImage(img)
        self.bottom_thumbnails.append(photo)  # Prevent GC
        label = ttk.Label(self.bottom_frame, image=photo)
        label.pack(side=tk.LEFT, padx=2)

    def clear_composed_level(self):
        self.added_image_indexes.clear()
        self.bottom_thumbnails.clear()
        for widget in self.bottom_frame.winfo_children():
            widget.destroy()

    def merge_selected_scenes(self):
        scenes = [self.generated_scenes[i] for i in self.added_image_indexes]
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
            level = self.get_sample_output(scene)
            level.play()

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
                level = self.get_sample_output(scene)
                level.save(file_path)
                print(f"Composed level saved to {file_path}")
            else:
                print("Save operation cancelled.")
        else:
            print("No composed scene to save.")

    def astar_composed_level(self):
        scene = self.merge_selected_scenes()
        if scene:
            level = self.get_sample_output(scene)
            level.run_astar()

    def get_sample_output(self, idx_or_scene):
        if isinstance(idx_or_scene, int):
            tensor = torch.tensor(self.current_levels[idx_or_scene])
            tile_numbers = torch.argmax(tensor, dim=0).numpy()
            char_grid = []
            for row in tile_numbers:
                char_row = "".join([self.id_to_char[num] for num in row])
                char_grid.append(char_row)
            level = SampleOutput(level=char_grid)
            return level
        else:
            # Assume idx_or_scene is a scene (list of lists of tile indices)
            char_grid = []
            for row in idx_or_scene:
                char_row = "".join([self.id_to_char[num] for num in row])
                char_grid.append(char_row)
            level = SampleOutput(level=char_grid)
            return level
      
    def play_level(self, idx):
        level = self.get_sample_output(idx)
        level.play()

    def use_astar(self, idx):
        level = self.get_sample_output(idx)
        level.run_astar()
  
root = tk.Tk()
app = CaptionBuilder(root)

global tileset_path
tileset_path = '..\TheVGLC\Super Mario Bros\smb.json'
if len(sys.argv) > 3:
    tileset_path = sys.argv[3]

if len(sys.argv) > 1:
    app.load_data(sys.argv[1])

if len(sys.argv) > 2:
    app.load_model(sys.argv[2])

root.mainloop()
