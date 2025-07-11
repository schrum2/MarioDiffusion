import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
import json
import torch
import gc
from PIL import ImageTk
import sys
from util.gui_shared import ParentBuilder, GUI_FONT_SIZE
from level_dataset import visualize_samples, convert_to_level_format, positive_negative_caption_split
from util.sampler import SampleOutput
from captions.caption_match import compare_captions
from captions.LR_caption_match import compare_captions as lr_compare_captions
from captions.MM_caption_match import compare_captions as mm_compare_captions
from create_ascii_captions import assign_caption
from LR_create_ascii_captions import assign_caption as lr_assign_caption
from MM_create_ascii_captions import assign_caption as mm_assign_caption
from captions.util import extract_tileset
import util.common_settings as common_settings
from util.sampler import scene_to_ascii
from models.pipeline_loader import get_pipeline
from level_dataset import append_absence_captions, remove_duplicate_phrases
from captions.caption_match import TOPIC_KEYWORDS
from ascii_data_browser import TileViewer
from models.fdm_pipeline import FDMPipeline


# Add the parent directory to sys.path so sibling folders can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

global tileset_path
tileset_path = None  # Global variable for tileset path
global game_selected
game_selected = None  # Global variable for selected game

# Global constant for GUI font size

GUI_FONT = ("Arial", GUI_FONT_SIZE)

class CaptionBuilder(ParentBuilder):
    global tileset_path, game_selected
    def __init__(self, master):
        global tileset_path, game_selected
        super().__init__(master) 
        # Set ttk style for font size
        style = ttk.Style()
        style.configure("TLabel", font=GUI_FONT)
        style.configure("TButton", font=GUI_FONT)
        style.configure("TCheckbutton", font=GUI_FONT)
        style.configure("TEntry", font=GUI_FONT)
        style.configure("TCombobox", font=GUI_FONT)
        
        # Holds tensors of levels currently on display
        self.current_levels = []
        self.generated_images = []
        self.generated_scenes = []

        # For tracking composed scenes and thumbnails
        self.composed_scenes = []
        self.composed_thumbnails = []
        self.composed_thumbnail_labels = []
        self.selected_composed_index = None
        self.present_caption = ""
        self.last_present_caption = ""

        # Frame for caption display
        self.caption_frame = ttk.Frame(master, width=200, borderwidth=2, relief="solid")  # Add border
        self.caption_frame.pack(side=tk.LEFT, fill=tk.Y, expand=False)  # Only fill vertically, don't expand horizontally
        
        self.caption_label = ttk.Label(self.caption_frame, text="Constructed Caption:", style="TLabel", font=GUI_FONT)
        self.caption_label.pack(pady=5)
        
        self.caption_text = tk.Text(self.caption_frame, height=8, state=tk.NORMAL, wrap=tk.WORD, font=GUI_FONT)
        self.caption_text.pack() 
                
        self.negative_prompt_label = ttk.Label(self.caption_frame, text="Negative Prompt:", style="TLabel")
        self.negative_prompt_label.pack()
        self.negative_prompt_entry = tk.Text(self.caption_frame, height=4, wrap=tk.WORD, font=GUI_FONT)
        self.negative_prompt_entry.pack()
        self.negative_prompt_entry.insert("1.0", "")

        self.automatic_negative_caption = tk.BooleanVar(value=False)
        self.automatic_negative_caption_checkbox = ttk.Checkbutton(self.caption_frame, text="Automatic Negative Captions", variable=self.automatic_negative_caption, style="TCheckbutton", command=self.update_negative_prompt_entry)
        self.automatic_negative_caption_checkbox.pack()
        
        # Automatic absence captions box
        self.automatic_absence_caption = tk.BooleanVar(value=False)
        self.automatic_absence_caption_checkbox = ttk.Checkbutton(self.caption_frame, text ="Automatic Absence Captions", variable=self.automatic_absence_caption, style="TCheckbutton", command=self.update_absence_caption_entry)
        self.automatic_absence_caption_checkbox.pack()
        self.automatic_absence_caption_checkbox.config(state=tk.DISABLED) # Start with the box disabled
        
        self.num_images_label = ttk.Label(self.caption_frame, text="Number of Images:", style="TLabel")
        self.num_images_label.pack()        
        self.num_images_entry = ttk.Entry(self.caption_frame, font=GUI_FONT)
        self.num_images_entry.pack()
        self.num_images_entry.insert(0, "4")

        self.seed_label = ttk.Label(self.caption_frame, text="Random Seed:", style="TLabel")
        self.seed_label.pack()        
        self.seed_entry = ttk.Entry(self.caption_frame, font=GUI_FONT)
        self.seed_entry.pack()
        self.seed_entry.insert(0, "1")

        self.num_steps_label = ttk.Label(self.caption_frame, text="Num Inference Steps:", style="TLabel")
        self.num_steps_label.pack()
        self.num_steps_entry = ttk.Entry(self.caption_frame, font=GUI_FONT)
        self.num_steps_entry.pack()
        self.num_steps_entry.insert(0, f"{common_settings.NUM_INFERENCE_STEPS}")
        
        self.guidance_label = ttk.Label(self.caption_frame, text="Guidance Scale:", style="TLabel")
        self.guidance_label.pack()
        self.guidance_entry = ttk.Entry(self.caption_frame, font=GUI_FONT)
        self.guidance_entry.pack()
        self.guidance_entry.insert(0, f"{common_settings.GUIDANCE_SCALE}")

        self.width_label = ttk.Label(self.caption_frame, text="Width (in tiles):", style="TLabel")
        self.width_label.pack()
        self.width_entry = ttk.Entry(self.caption_frame, font=GUI_FONT)
        self.width_entry.pack()
        self.height_label = ttk.Label(self.caption_frame, text="Height (in tiles):")
        self.height_label.pack()
        self.height_entry = ttk.Entry(self.caption_frame, font=GUI_FONT)
        self.height_entry.pack()
        if game_selected == "Lode Runner":
            self.width_entry.insert(0, f"{common_settings.LR_WIDTH}")
            self.height_entry.insert(0, f"{common_settings.LR_HEIGHT}")
        elif game_selected == "Mario":
            self.width_entry.insert(0, f"{common_settings.MARIO_WIDTH}")
            self.height_entry.insert(0, f"{common_settings.MARIO_HEIGHT}")
        else:
            self.width_entry.insert(0, f"{common_settings.MEGAMAN_WIDTH}")
            self.height_entry.insert(0, f"{common_settings.MEGAMAN_HEIGHT}")

        self.generate_button = ttk.Button(self.caption_frame, text="Generate Image", command=self.generate_image)
        self.generate_button.pack(pady=5)
                
        self.model_button = ttk.Button(self.checkbox_frame, text="Load Model", command=self.load_model, style="TButton")
        self.model_button.pack(anchor=tk.E)

        self.uncheck_all_button = ttk.Button(self.checkbox_frame, text="Uncheck All", command=self.uncheck_all)
        self.uncheck_all_button.pack(anchor=tk.E)

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

        #Bind mousewheel scrolling globally, and scroll the widget under the mouse if it's a canvas
        self.master.bind_all("<MouseWheel>", self._on_mousewheel)

        self.checkbox_vars = {}

        self.loaded_model_label = ttk.Label(self.caption_frame, text=f"Using model: Not loaded yet", style="TLabel")
        self.loaded_model_label.pack()

        self.debug_caption = tk.BooleanVar(value=False)
        self.debug_caption_checkbox = ttk.Checkbutton(self.caption_frame, text="Debug Caption Match", variable=self.debug_caption, style="TCheckbutton")
        self.debug_caption_checkbox.pack()

        # Frame for composed level controls
        self.composed_frame = ttk.Frame(self.caption_frame)
        self.composed_frame.pack(fill=tk.X, pady=(20, 5))  # 20 pixels above, 5 below

        # First row: Checkbox, Play, Use A*
        row1 = ttk.Frame(self.composed_frame)
        row1.pack(pady=(10, 0), anchor="center")
        # Second row: Delete, Clear, Save
        row2 = ttk.Frame(self.composed_frame)
        row2.pack(pady=(10, 0), anchor="center")
        # Third row: Move selection left/right
        row3 = ttk.Frame(self.composed_frame)
        row3.pack(pady=(10, 0), anchor="center")

        self.play_composed_button = ttk.Button(row1, text="Play Composed Level", command=self.play_composed_level, style="TButton")
        self.play_composed_button.pack(side=tk.LEFT, padx=5)
        self.astar_composed_button = ttk.Button(row1, text="Use A* on Composed Level", command=self.astar_composed_level, style="TButton")
        self.astar_composed_button.pack(side=tk.LEFT, padx=5)
        self.use_snes_graphics = tk.BooleanVar(value=False)
        self.graphics_checkbox = ttk.Checkbutton(row1, text="Use SNES Graphics", variable=self.use_snes_graphics, style="TCheckbutton")
        self.graphics_checkbox.pack(side=tk.LEFT, padx=5)

        self.delete_image_button = ttk.Button(row2, text="Delete Selected Image", command=self.delete_selected_composed_image, style="TButton")
        self.delete_image_button.pack(side=tk.LEFT, padx=10)
        self.clear_composed_button = ttk.Button(row2, text="Clear Composed Level", command=self.clear_composed_level, style="TButton")
        self.clear_composed_button.pack(side=tk.LEFT, padx=10)
        self.save_composed_button = ttk.Button(row2, text="Save Composed Level", command=self.save_composed_level, style="TButton")
        self.save_composed_button.pack(side=tk.LEFT, padx=10)
        
        self.move_left_button = ttk.Button(row3, text="Move Selected Image Left", command=lambda: self.move_selected_image(-1), style="TButton")
        self.move_left_button.pack(side=tk.LEFT, padx=60)
        self.move_right_button = ttk.Button(row3, text="Move Selected Image Right", command=lambda: self.move_selected_image(1), style="TButton")
        self.move_right_button.pack(side=tk.LEFT, padx=60)

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
        self.game_label = ttk.Label(self.caption_frame, text="Select Game:", style="TLabel")
        self.game_label.pack()
        self.game_dropdown = ttk.Combobox(self.caption_frame, textvariable=self.game_var, values=["Mario", "Lode Runner", "Mega Man (Simple)", "Mega Man (Full)"], state="readonly", font=GUI_FONT)
        self.game_dropdown.pack()
        
    def probe_absence_caption_support(self):
        """Test if the loaded model supports absence captions by running a quick, hidden generation."""
        try:
            # Use a minimal absence caption prompt
            test_prompt = append_absence_captions("", TOPIC_KEYWORDS)
            # Minimal params for a fast test
            param_values = {
                "num_inference_steps": 1,
                "guidance_scale": 1.0,
                "width": 4,
                "height": 4,
                "output_type": "tensor",
                "caption": test_prompt
            }
            generator = torch.Generator(self.device).manual_seed(1)
            # Try generating (do not display or store result)
            _ = self.pipe(generator=generator, **param_values)
            # If no exception, enable the checkbox
            self.automatic_absence_caption_checkbox.config(state=tk.NORMAL)
        except Exception as e:
            # If any error, disable the checkbox
            self.automatic_absence_caption_checkbox.config(state=tk.DISABLED)
            self.automatic_absence_caption.set(False)

    def probe_diffusion_args_support(self):
        """Test if the loaded model can use our diffusion-specific args, greys them out if it can't"""
        if isinstance(self.pipe, FDMPipeline):
            #We're using an FDM model here, so we remove support for negative prompts, guidance scale, inference steps, and control over the width/height of the output.
            self.negative_prompt_entry.delete("1.0", tk.END)
            self.negative_prompt_entry.config(state=tk.DISABLED)

            self.automatic_negative_caption.set(False)
            self.automatic_negative_caption_checkbox.config(state=tk.DISABLED)

            self.guidance_entry.config(state=tk.DISABLED)

            self.num_steps_entry.config(state=tk.DISABLED)

            self.width_entry.config(state=tk.DISABLED)

            self.height_entry.config(state=tk.DISABLED)
        else:
            #If this isn't the case, return everything back to normal
            self.negative_prompt_entry.config(state=tk.NORMAL)

            self.automatic_negative_caption_checkbox.config(state=tk.NORMAL)

            self.guidance_entry.config(state=tk.NORMAL)

            self.num_steps_entry.config(state=tk.NORMAL)

            self.width_entry.config(state=tk.NORMAL)

            self.height_entry.config(state=tk.NORMAL)


    def create_image_context_menu(self, pil_image, image_index):
        """Create a context menu for right-clicking on images"""
        context_menu = tk.Menu(self.master, tearoff=0)
        context_menu.add_command(
            label="Save Image As...", 
            command=lambda: self.save_image_as(pil_image, image_index)
        )
        return context_menu

    def show_context_menu(self, event, context_menu):
        """Show the context menu at the cursor position"""
        try:
            context_menu.tk_popup(event.x_root, event.y_root)
        finally:
            context_menu.grab_release()

    def save_image_as(self, pil_image, image_index):
        """Save the PIL image to a file chosen by the user"""
        # Create default filename
        default_filename = f"generated_level_{image_index + 1}.png"
        
        # Open save dialog
        file_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[
                ("PNG files", "*.png"),
                ("JPEG files", "*.jpg"),
                ("All files", "*.*")
            ],
            title="Save Image As",
            initialfile=default_filename  # Changed from initialfilename to initialfile
        )
        
        if file_path:
            try:
                # Save the image
                pil_image.save(file_path)
                messagebox.showinfo("Success", f"Image saved successfully to:\n{file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save image:\n{str(e)}")

    def get_patterns(self):
        # Different for LoRA and tile diffusion
        patterns = [
                    # Mario and Lode Runner patterns
                    "floor", "ceiling", "platform", 
                    "rectangular", "irregular", "enem",

                    # Lode Runner patterns
                    "ladder", "gold", "rope",
                    "chamber", "background area",
                    "diggable ground", "solid ground",

                    # Mario patterns
                    "pipe", "coin", "tower", #"wall",
                    "cannon", "staircase", 
                    "question block", "loose block", 
                    
                    #Mega Man phrases
                    "entrance direction", "exit direction",
                    "powerup", "hazard", "water",
                    "dissapearing block"
                    ]
        return patterns

    def load_data(self, filepath = None):
        global tileset_path, game_selected
        if filepath == None:
            filepath = filedialog.askopenfilename(title="Select JSON File", filetypes=[("JSON", "*.json")])
        if filepath:
            _, self.id_to_char, self.char_to_id, self.tile_descriptors = extract_tileset(tileset_path)
            # print(f"Tileset in use: {tileset_path}")
            # print(f"Self ID to Char: {self.id_to_char}")
            # print(f"Self Char to ID: {self.char_to_id}")
            # print(f"Self Tile Descriptors: {self.tile_descriptors}")

            try:
                phrases_set = set()
                with open(filepath, 'r') as f:
                    dataset = json.load(f)
                    for item in dataset:
                        phrases = item['caption'].split('.')
                        phrases_set.update(phrase.strip() for phrase in phrases if phrase.strip())
                        if self.automatic_absence_caption.get():
                            self.update_absence_caption_entry()
                        if self.automatic_negative_caption.get():
                            self.update_negative_prompt_entry
                            
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
            self.pipe = get_pipeline(model).to(self.device)
            
            # Probe for absence caption support before updating GUI
            self.probe_absence_caption_support()

            # Probe to grey out diffusion args if we're using the FDM model
            self.probe_diffusion_args_support()

            filename = os.path.splitext(os.path.basename(model))[0]
            self.loaded_model_label["text"] = f"Using model: {filename}"
    
            # Enable or disable negative prompt entry based on pipeline support
            if hasattr(self.pipe, "supports_negative_prompt") and self.pipe.supports_negative_prompt:
                self.negative_prompt_entry.config(state=tk.NORMAL)
                self.automatic_negative_caption_checkbox.config(command=self.update_negative_prompt_entry)
            else:
                self.negative_prompt_entry.delete("1.0", tk.END)
                self.negative_prompt_entry.config(state=tk.DISABLED)
                self.automatic_negative_caption_checkbox.config(state=tk.DISABLED)

    def update_caption(self):
        self.selected_phrases = [phrase for phrase, var in self.checkbox_vars.items() if var.get()]
        self.present_caption = ". ".join(self.selected_phrases) + "." if self.selected_phrases else ""
        self.last_present_caption = self.present_caption  # Save for absence toggling

        if self.automatic_absence_caption.get():
            # Only use the currently checked phrases as the present caption
            cleaned_prompt = self.present_caption
            self.last_present_caption = cleaned_prompt
            absence_caption = append_absence_captions(cleaned_prompt, TOPIC_KEYWORDS)
            absence_caption = remove_duplicate_phrases(absence_caption)
            self.caption_text.config(state=tk.NORMAL)
            self.caption_text.delete(1.0, tk.END)
            self.caption_text.insert(tk.END, absence_caption)
            self.caption_text.config(state=tk.NORMAL)
        else:
            self.caption_text.config(state=tk.NORMAL)
            self.caption_text.delete(1.0, tk.END)
            self.caption_text.insert(tk.END, self.present_caption)
            self.caption_text.config(state=tk.NORMAL)

        if self.automatic_negative_caption.get():
            # Only use the currently checked phrases as the present caption
            cleaned_neg_prompt = self.present_caption
            self.last_present_neg_caption = cleaned_neg_prompt
            pos, neg = positive_negative_caption_split(self.last_present_neg_caption, True)
            negative_caption = remove_duplicate_phrases(neg)
            self.negative_prompt_entry.config(state=tk.NORMAL)
            self.negative_prompt_entry.delete(1.0, tk.END)
            self.negative_prompt_entry.insert(tk.END, negative_caption)
            self.negative_prompt_entry.config(state=tk.NORMAL)
        else:
            self.negative_prompt_entry.config(state=tk.NORMAL)
            self.negative_prompt_entry.delete(1.0, tk.END)
            #self.negative_prompt_entry.insert(tk.END, self.present_caption)
            self.negative_prompt_entry.config(state=tk.NORMAL)
    
    def generate_image(self):
        global tileset_path, game_selected
        game_selected = self.game_var.get()
        # # cannot use multiple generations of levels in one composed level
        # self.clear_composed_level()
        # print("Clearing previously composed level for newly generated scenes.")

        # clear the previous images
        self.generated_images = []
        self.generated_scenes = []

        print("Generating")
        
        if self.automatic_absence_caption.get():
            prompt = append_absence_captions(self.caption_text.get("1.0", tk.END).strip(), TOPIC_KEYWORDS)
        else:
            prompt = self.caption_text.get("1.0", tk.END).strip()
        
        #prompt = self.caption_text.get("1.0", tk.END).strip()
        # prompt = self.present_caption.strip()
        negative_prompt = self.negative_prompt_entry.get("1.0", tk.END).strip()
        num_images = int(self.num_images_entry.get())        
        param_values = {
            "num_inference_steps": int(self.num_steps_entry.get()),
            "guidance_scale": float(self.guidance_entry.get()),
            "width": int(self.width_entry.get()),
            "height": int(self.height_entry.get()),
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
            if "caption" in param_values: print(f"Caption: {param_values['caption']}")
            else: print("No caption")
            images = self.pipe(generator=generator, **param_values).images
            self.current_levels.append(images[0].cpu().detach().numpy())

            sample_tensor = images[0].unsqueeze(0)
            sample_indices = convert_to_level_format(sample_tensor)
            #print("images:", images)
            scene = sample_indices[0].tolist()

            if game_selected == "Lode Runner":
                number_of_tiles = common_settings.LR_TILE_COUNT
                scene = [[x % number_of_tiles for x in row] for row in scene]
                tileset_path = common_settings.LR_TILESET
                _, self.id_to_char, self.char_to_id, self.tile_descriptors = extract_tileset(tileset_path)
            elif game_selected == "Mega Man (Simple)":
                number_of_tiles = common_settings.MM_SIMPLE_TILE_COUNT
                scene = [[x % number_of_tiles for x in row] for row in scene]
                tileset_path = common_settings.MM_SIMPLE_TILESET
                _, self.id_to_char, self.char_to_id, self.tile_descriptors = extract_tileset(tileset_path)
            elif game_selected == "Mega Man (Full)":
                number_of_tiles = common_settings.MM_FULL_TILE_COUNT
                scene = [[x % number_of_tiles for x in row] for row in scene]
                tileset_path = common_settings.MM_FULL_TILESET
                _, self.id_to_char, self.char_to_id, self.tile_descriptors = extract_tileset(tileset_path)
            
            self.generated_scenes.append(scene)
            #selected_game = self.game_var.get()
            if game_selected == "Lode Runner":
                actual_caption = lr_assign_caption(scene, self.id_to_char, self.char_to_id, self.tile_descriptors, False, False)
                pil_img = visualize_samples(images, game='LR')
            elif game_selected == "Mario":
                actual_caption = assign_caption(scene, self.id_to_char, self.char_to_id, self.tile_descriptors, False, False)
                pil_img = visualize_samples(images)
            else:
                actual_caption = mm_assign_caption(scene, self.id_to_char, self.char_to_id, self.tile_descriptors, False, False)
                pil_img = visualize_samples(images, game="MM-Simple" if game_selected == "Mega Man (Simple)" else "MM-Full")

            self.generated_images.append(pil_img)
            img_tk = ImageTk.PhotoImage(pil_img)
            if game_selected == 'Mario':
                compare_score, exact_matches, partial_matches, excess_phrases = compare_captions(prompt, actual_caption, return_matches=True, debug=self.debug_caption.get())
            elif game_selected == 'Lode Runner':
                compare_score, exact_matches, partial_matches, excess_phrases = lr_compare_captions(prompt, actual_caption, return_matches=True, debug=self.debug_caption.get())
            else:
                compare_score, exact_matches, partial_matches, excess_phrases = mm_compare_captions(prompt, actual_caption, return_matches=True, debug=self.debug_caption.get())

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

            # Create context menu for this image
            context_menu = self.create_image_context_menu(pil_img, i)

            # Bind right-click to show context menu
            label.bind("<Button-3>", lambda event, menu=context_menu: self.show_context_menu(event, menu))
            # For macOS compatibility, also bind Control+Click
            label.bind("<Control-Button-1>", lambda event, menu=context_menu: self.show_context_menu(event, menu))

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
            if game_selected == "Mario":
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
            elif game_selected == "Lode Runner":
                if len(scene[0]) > common_settings.LR_WIDTH:
                    from captions.LR_caption_match import process_scene_segments as lr_process_scene_segments
                    avg_segment_score, _, _ = lr_process_scene_segments(
                        scene=scene,
                        segment_width=common_settings.LR_WIDTH,
                        prompt=prompt,
                        id_to_char=self.id_to_char,
                        char_to_id=self.char_to_id,
                        tile_descriptors=self.tile_descriptors,
                        describe_locations=False,
                        describe_absence=False
                    )
            else:
                if len(scene[0]) > common_settings.MEGAMAN_WIDTH:
                    from captions.MM_caption_match import process_scene_segments as mm_process_scene_segments
                    avg_segment_score, _, _ = mm_process_scene_segments(
                        scene=scene,
                        segment_width=common_settings.LR_WIDTH,
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
                command=lambda idx=i: self.play_level(idx),
                style="TButton"
            )
            play_button.pack(side=tk.LEFT, padx=5)
    
            # Add Use A* button
            astar_button = ttk.Button(
                button_frame, 
                text="Use A*", 
                command=lambda idx=i: self.use_astar(idx),
                style="TButton"
            )
            astar_button.pack(side=tk.LEFT, padx=5)

            # Add "Add To Level" button
            add_button = ttk.Button(
                button_frame,
                text="Add To Level",
                command=lambda idx=i: self.add_to_composed_level(idx),
                style="TButton"
            )
            add_button.pack(side=tk.LEFT, padx=5)

            del images, sample_tensor, sample_indices, scene  # Delete unused tensors
            torch.cuda.empty_cache()  # Clear the cache
            gc.collect()  # Force garbage collection

        print("Image generation completed.")
        #print(self.current_levels)

    def add_to_composed_level(self, idx):
        # Store the actual scene
        scene = self.generated_scenes[idx]
        if game_selected == "Lode Runner":
                number_of_tiles = common_settings.LR_TILE_COUNT
                scene = [[x % number_of_tiles for x in row] for row in scene]
                tileset_path = common_settings.LR_TILESET
        elif game_selected == "Mega Man (Simple)":
                number_of_tiles = common_settings.MM_SIMPLE_TILE_COUNT
                scene = [[x % number_of_tiles for x in row] for row in scene]
                tileset_path = common_settings.MM_SIMPLE_TILESET
        elif game_selected == "Mega Man (Full)":
            number_of_tiles = common_settings.MM_FULL_TILE_COUNT
            scene = [[x % number_of_tiles for x in row] for row in scene]
            tileset_path = common_settings.MM_FULL_TILESET
        self.composed_scenes.append(scene)

        # Create and store the thumbnail
        img = self.generated_images[idx].copy()
        img.thumbnail((64, 64))
        photo = ImageTk.PhotoImage(img)
        self.composed_thumbnails.append(photo)  # Prevent GC

        # Create a clickable label for the thumbnail
        label = ttk.Label(self.bottom_frame, image=photo, borderwidth=2, relief="flat")
        label.pack(side=tk.LEFT, padx=2)
        self.composed_thumbnail_labels.append(label)
        self.rebind_composed_thumbnail_clicks()

    def select_composed_thumbnail(self, index):
        # Deselect all
        for lbl in self.composed_thumbnail_labels:
            lbl.config(relief="flat", borderwidth=2)
        # Select the clicked one
        self.composed_thumbnail_labels[index].config(relief="solid", borderwidth=4)
        self.selected_composed_index = index

    def rebind_composed_thumbnail_clicks(self):
        """
        Updates the click event bindings for each thumbnail label to ensure 
        that when you click a thumbnail, the correct index is assigned
        This must be called after any operation that changes the order,
        adds, or removes thumbnails, to keep selection working correctly.
        """
        for i, lbl in enumerate(self.composed_thumbnail_labels):
            lbl.bind("<Button-1>", lambda e, i=i: self.select_composed_thumbnail(i))

    def delete_selected_composed_image(self):
        idx = self.selected_composed_index
        if idx is not None and 0 <= idx < len(self.composed_scenes):
            # Remove from all lists
            self.composed_scenes.pop(idx)
            self.composed_thumbnails.pop(idx)
            label = self.composed_thumbnail_labels.pop(idx)
            label.destroy()
            self.selected_composed_index = None
            # Rebind click events for all remaining labels
            self.rebind_composed_thumbnail_clicks()
        else:
            messagebox.showinfo("No selection", "Please select a thumbnail first.")

    def move_selected_image(self, direction):
        idx = self.selected_composed_index
        if idx is None or not (0 <= idx < len(self.composed_scenes)):
            messagebox.showinfo("No selection", "Please select a thumbnail first.")
            return

        new_idx = idx + direction
        if not (0 <= new_idx < len(self.composed_scenes)):
            return  # Out of bounds, do nothing

        # Swap in all lists
        for lst in [self.composed_scenes, self.composed_thumbnails, self.composed_thumbnail_labels]:
            lst[idx], lst[new_idx] = lst[new_idx], lst[idx]

        # Remove all labels and re-pack in new order
        for lbl in self.composed_thumbnail_labels:
            lbl.pack_forget()
        for lbl in self.composed_thumbnail_labels:
            lbl.pack(side=tk.LEFT, padx=2)

        # Rebind click events with correct indices
        self.rebind_composed_thumbnail_clicks()

        # Update selection
        self.select_composed_thumbnail(new_idx)

    def clear_composed_level(self):
        self.composed_scenes.clear()
        self.composed_thumbnails.clear()
        self.composed_thumbnail_labels.clear()
        self.selected_composed_index = None
        for widget in self.bottom_frame.winfo_children():
            widget.destroy()

    def merge_selected_scenes(self):
        scenes = self.composed_scenes
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
            level = self.get_sample_output(scene, use_snes_graphics=self.use_snes_graphics.get())
            console_output = level.run_astar()
            print(console_output)

    def get_sample_output(self, idx_or_scene, use_snes_graphics=False):
        if isinstance(idx_or_scene, int):
            tensor = torch.tensor(self.current_levels[idx_or_scene])
            tile_numbers = torch.argmax(tensor, dim=0).numpy()
            if game_selected == "Lode Runner":
                tile_numbers = [[int(num) % len(self.id_to_char) for num in row] for row in tile_numbers]
                #char_grid = scene_to_ascii(tile_numbers, self.id_to_char, shorten=False)
                level = SampleOutput(level=tile_numbers, use_snes_graphics=use_snes_graphics)
            else:
                char_grid = scene_to_ascii(tile_numbers, self.id_to_char)
                level = SampleOutput(level=char_grid, use_snes_graphics=use_snes_graphics)
            return level
        else:
            # Assume idx_or_scene is a scene (list of lists of tile indices)
            if game_selected == "Lode Runner":
                tile_numbers = [[int(num) % len(self.id_to_char) for num in row] for row in tile_numbers]
                char_grid = scene_to_ascii(tile_numbers, self.id_to_char, shorten=False)
            else:
                char_grid = scene_to_ascii(tile_numbers, self.id_to_char)
            level = SampleOutput(level=char_grid, use_snes_graphics=use_snes_graphics)
            return level
      
    def play_level(self, idx):
        selected_game = self.game_var.get()
        if selected_game == "Lode Runner":
            import tempfile, json
            level = self.get_sample_output(idx, use_snes_graphics=self.use_snes_graphics.get())
            #print("Level to play:", level)
            level.play(game="loderunner", level_idx=1)
        else:
            #Default: Mario play logic
            level = self.get_sample_output(idx, use_snes_graphics=self.use_snes_graphics.get())
            level.play()

    def use_astar(self, idx):
        level = self.get_sample_output(idx, use_snes_graphics=self.use_snes_graphics.get())
        console_output = level.run_astar()
        print(console_output)

    def uncheck_all(self):
        """Uncheck all checkboxes in the provided list or dict."""
        for var in self.checkbox_vars.values():
            var.set(0)
            self.update_caption()

    def _on_mousewheel(self, event):
        """Handle mouse wheel scrolling for both image and checkbox canvases."""
        widget_under_mouse = self.master.winfo_containing(event.x_root, event.y_root)
        # Check if widget_under_mouse is self.image_canvas or a descendant
        parent = widget_under_mouse
        while parent is not None:
            if parent == self.image_canvas:
                self.image_canvas.yview_scroll(-1 * (event.delta // 120), "units")
                break
            elif parent == self.checkbox_canvas:
                self.checkbox_canvas.yview_scroll(-1 * (event.delta // 120), "units")
            parent = parent.master

    def update_absence_caption_entry(self):
        """Update the constructed caption box based on the absence caption checkbox."""
        if self.automatic_absence_caption.get():
            # Remove all "no ..." phrases from the current box
            current_text = self.caption_text.get("1.0", tk.END).strip()
            cleaned_phrases = [phrase.strip() for phrase in current_text.split('.') if phrase.strip() and "no" not in phrase]
            cleaned_prompt = ". ".join(cleaned_phrases)
            if cleaned_prompt:
                cleaned_prompt += "."
            self.last_present_caption = cleaned_prompt
            absence_caption = append_absence_captions(cleaned_prompt, TOPIC_KEYWORDS)
            self.caption_text.config(state=tk.NORMAL)
            self.caption_text.delete(1.0, tk.END)
            self.caption_text.insert(tk.END, absence_caption)
            self.caption_text.config(state=tk.NORMAL)
        else:
            self.caption_text.config(state=tk.NORMAL)
            self.caption_text.delete(1.0, tk.END)
            self.caption_text.insert(tk.END, self.last_present_caption)
            self.caption_text.config(state=tk.NORMAL)

    def update_negative_prompt_entry(self):
        """Update the negative prompt entry based on the automatic negative caption checkbox."""
        if self.automatic_negative_caption.get():
            current_text = self.caption_text.get("1.0", tk.END).strip()
            cleaned_neg_phrases = [phrase.strip() for phrase in current_text.split('.') if phrase.strip()]
            cleaned_neg_prompt = ". ".join(cleaned_neg_phrases)
            if cleaned_neg_prompt:
                cleaned_neg_prompt += "."
            self.last_present_caption = cleaned_neg_prompt
            pos, neg = positive_negative_caption_split(self.last_present_caption, True)
            self.negative_prompt_entry.delete("1.0", tk.END)
            self.negative_prompt_entry.insert("1.0", neg)
            # Disable the entry if automatic negative caption is checked
            self.negative_prompt_entry.config(state=tk.DISABLED)
        else:
            self.negative_prompt_entry.config(state=tk.NORMAL)
            self.negative_prompt_entry.delete(1.0, tk.END)
            #self.negative_prompt_entry.insert(tk.END, self.last_present_neg_caption)
            self.negative_prompt_entry.config(state=tk.NORMAL)


import argparse
def parse_args():
    parser = argparse.ArgumentParser(description="Interactive Tile Level Generator")
    parser.add_argument(
        "--game",
        type=str,
        default="Mario",
        choices=["Mario", "LR"],
        help="Which game to create a model for (affects sample style and tile count)"
    )
    parser.add_argument("--model_path", type=str, help="Path to the trained diffusion model")
    parser.add_argument("--load_data", type=str, default="datasets/Mar1and2_LevelsAndCaptions-regular.json", help="Path to the dataset JSON file")
    parser.add_argument("--tileset", default=common_settings.MARIO_TILESET, help="Descriptions of individual tile types")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    if args.game == "Mario":
        game_selected = "Mario"
        tileset_path = common_settings.MARIO_TILESET
    elif args.game == "LR":
        game_selected = "Lode Runner"
        tileset_path = common_settings.LR_TILESET

    root = tk.Tk()
    app = CaptionBuilder(root)
    app.load_data(args.load_data)
    app.load_model(args.model_path)

    root.mainloop()
