import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
import json
import torch
import gc
from PIL import ImageTk, Image
import sys
from util.gui_shared import ParentBuilder, GUI_FONT_SIZE
from level_dataset import visualize_samples, convert_to_level_format, positive_negative_caption_split, mario_tiles, lr_tiles, mm_tiles
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
        self.generated_widget_refs = []


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

        # TODO: Currently the code produces errors if you try to generate images when the wrong game is selected.
        # Instead of printing a stack trace error in the console, make the code produce a friendly pop-up error 
        # indicating that the wrong game might be selected.

                
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

        self.show_astar_var = tk.BooleanVar(value=False)
        self.show_astar_checkbox = ttk.Checkbutton(
            self.caption_frame,
            text="With Simple A*",
            variable=self.show_astar_var,
            style="TCheckbutton",
            command=self.toggle_all_astar_overlays
        )
        self.show_astar_checkbox.pack()

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

        self.mm_layout_button = ttk.Button(row1, text="Arrange Mega Man Level", command=self.open_megaman_layout_editor, style="TButton")
        self.mm_layout_button.pack(side=tk.LEFT, padx=5)

        self.delete_image_button = ttk.Button(row2, text="Delete Selected Image", command=self.delete_selected_composed_image, style="TButton")
        self.delete_image_button.pack(side=tk.LEFT, padx=10)
        self.clear_composed_button = ttk.Button(row2, text="Clear Composed Level", command=self.clear_composed_level, style="TButton")
        self.clear_composed_button.pack(side=tk.LEFT, padx=10)
        self.save_composed_button = ttk.Button(row2, text="Save Composed Level", command=self.save_composed_level, style="TButton")
        self.save_composed_button.pack(side=tk.LEFT, padx=10)
        
        self.move_left_button = ttk.Button(row3, text="Move Selected Image Left", command=lambda: self.move_selected_image(-1), style="TButton")
        self.move_left_button.pack(side=tk.LEFT, padx=15)

        self.large_view_button = ttk.Button(row3, text="Large View", command=self.show_large_composed_view, style="TButton")
        self.large_view_button.pack(side=tk.LEFT, padx=15)

        self.move_right_button = ttk.Button(row3, text="Move Selected Image Right", command=lambda: self.move_selected_image(1), style="TButton")
        self.move_right_button.pack(side=tk.LEFT, padx=15)

        self.edit_composed_button = ttk.Button(row3, text="Edit Selected Image", command=self.edit_selected_composed_image, style="TButton")
        self.edit_composed_button.pack(side=tk.LEFT, padx=15)

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
        # Game selection
        self.game_var = tk.StringVar(value=game_selected if game_selected else "Mario")
        
        self.game_label = ttk.Label(self.caption_frame, text="Select Game:", style="TLabel")
        self.game_label.pack()
        self.game_dropdown = ttk.Combobox(self.caption_frame, textvariable=self.game_var, values=["Mario", "Lode Runner", "Mega Man (Simple)", "Mega Man (Full)"], state="readonly", font=GUI_FONT)
        self.game_dropdown.pack()
        self.game_dropdown.bind("<<ComboboxSelected>>", lambda e: self.update_mario_only_buttons()) 
        self.update_mario_only_buttons() 
        
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


    def _play_megaman_level(self, idx):
        import subprocess, os
        from util.sampler import scene_to_ascii
 
        scene = self.generated_scenes[idx]
        scene_to_ascii(scene, self.id_to_char, shorten=False)

        # Save as .txt first
        txt_path = os.path.join(os.getcwd(), "temp_mm_level.txt")
        with open(txt_path, 'w') as f:
            for row in char_grid:
                f.write(''.join(row) + '\n')

        # Convert to .mmlv
        mmlv_path = os.path.join(
            os.path.expanduser("~"),
            "AppData", "Local", "MegaMaker", "Levels", "generated_level.mmlv"
        )
        from megaman.vglc_to_mmlv import convert
        lines = open(txt_path).readlines()
        result = convert(lines, level_name="Generated", author="AI")
        with open(mmlv_path, 'w', encoding='utf-8', newline='\n') as f:
            f.write(result)

        # Open Mega Man Maker
        mm_exe = r"C:\Program Files (x86)\MegaMaker\MegaMaker.exe"
        if os.path.exists(mm_exe):
            subprocess.Popen([mm_exe])
        else:
            import tkinter.messagebox as mb
            mb.showinfo("Saved", f"Level saved to:\n{mmlv_path}\n\nCould not find MegaMaker.exe — open it manually.")

    def _play_megaman_level_from_scene(self, scene):
        import subprocess
        from util.sampler import scene_to_ascii

        char_grid = char_grid = scene_to_ascii(scene, self.id_to_char, shorten=False)


        txt_path = os.path.join(os.getcwd(), "temp_mm_level.txt")
        with open(txt_path, 'w') as f:
            for row in char_grid:
                f.write(''.join(row) + '\n')

        mmlv_path = os.path.join(
            os.path.expanduser("~"),
            "AppData", "Local", "MegaMaker", "Levels", "generated_level.mmlv"
        )
        from megaman.vglc_to_mmlv import convert
        lines = open(txt_path).readlines()
        result = convert(lines, level_name="Generated", author="AI")
        with open(mmlv_path, 'w', encoding='utf-8', newline='\n') as f:
            f.write(result)

        mm_exe = r"C:\Program Files (x86)\MegaMaker\MegaMaker.exe"
        if os.path.exists(mm_exe):
            subprocess.Popen([mm_exe])
        else:
            messagebox.showinfo("Saved", f"Level saved to:\n{mmlv_path}\n\nCould not find MegaMaker.exe — open it manually.")

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
                    "disappearing block"
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

            '''detected_game = self.detect_game_from_model()
            if detected_game:
                self.game_var.set(detected_game)
                self.update_mario_only_buttons()
                # Also update width/height entries to match
                self._apply_game_defaults(detected_game)'''
            
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

    # creates a pop-up window to ask the user to confirm if the detected game is correct when the tile count is ambiguous (e.g. 13 tiles could be either Mario or Mega)
    #but, we dont like how it pops up every time so for now were commenting it out and for future if yall can find a more elegant way to handle ambiguous tile counts that would be great
    '''  def detect_game_from_model(self):
            try:
                if hasattr(self.pipe, 'unet'):
                    tile_count = self.pipe.unet.config.out_channels
                elif hasattr(self.pipe, 'model'):
                    tile_count = self.pipe.model.config.out_channels  # adjust if FDM differs
                else:
                    return None  # can't detect

                if tile_count == common_settings.LR_TILE_COUNT:        # 8
                    return "Lode Runner"
                elif tile_count == common_settings.MM_FULL_TILE_COUNT:  # 41
                    return "Mega Man (Full)"
                elif tile_count == 13:
                    # Ambiguous — ask the user to confirm
                    answer = messagebox.askyesno(
                        "Confirm Game",
                        "Is this a Mario model?\n\n"
                        "Click Yes for Mario, No for Mega Man (Simple)."
                    )
                    return "Mario" if answer else "Mega Man (Simple)"
                else:
                    return None  # unknown tile count
            except Exception:
                return None  # silently fail if attributes aren't there

        def _apply_game_defaults(self, game):
            self.width_entry.config(state=tk.NORMAL)
            self.height_entry.config(state=tk.NORMAL)
            if game == "Lode Runner":
                w, h = common_settings.LR_WIDTH, common_settings.LR_HEIGHT
            elif game == "Mario":
                w, h = common_settings.MARIO_WIDTH, common_settings.MARIO_HEIGHT
            else:
                w, h = common_settings.MEGAMAN_WIDTH, common_settings.MEGAMAN_HEIGHT
            self.width_entry.delete(0, tk.END)
            self.width_entry.insert(0, str(w))
            self.height_entry.delete(0, tk.END)
            self.height_entry.insert(0, str(h)) '''

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

        self.generated_widget_refs = [] 

        print("Generating")
        
        prompt = self._get_current_prompt()
        
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
            try:
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

            except Exception as e:
                messagebox.showerror(
                    "Generation Error",
                    f"Failed to generate image {i + 1}.\n\n"
                    f"This may be caused by selecting the wrong game for the loaded model.\n\n"
                    f"Details: {str(e)}"
                )
                break

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

            self.generated_widget_refs.append({
                "image_label": label,
                "caption_text": caption_text,
                "score_label": score_label,
            })

            # Create a frame for buttons
            button_frame = ttk.Frame(img_frame)
            button_frame.pack(pady=5)
    
            is_mario = game_selected == "Mario"

            # Add Play button
            play_button = ttk.Button(
                button_frame, 
                text="Play", 
                command=lambda idx=i: self.play_level(idx),
                style="TButton",
                state=tk.NORMAL if is_mario else tk.DISABLED 

            )
            play_button.pack(side=tk.LEFT, padx=5)
    
            # Add Use A* button
            astar_button = ttk.Button(
                button_frame,
                text="Use A*",
                command=lambda idx=i: self.use_astar(idx),
                style="TButton",
                state=tk.NORMAL if is_mario else tk.DISABLED
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

            edit_button = ttk.Button(
                button_frame,
                text="Edit",
                command=lambda idx=i: self.edit_level(idx),
                style="TButton"
            )
            edit_button.pack(side=tk.LEFT, padx=5)

            if self.show_astar_var.get():
                self._show_astar_overlay_for_index(i)

            del images, sample_tensor, sample_indices, scene  # Delete unused tensors
            if torch.cuda.is_available():
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

    def edit_selected_composed_image(self):
        idx = self.selected_composed_index
        if idx is None or not (0 <= idx < len(self.composed_scenes)):
            messagebox.showinfo("No selection", "Please select a thumbnail first.")
            return
        self.edit_composed_scene(idx)

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

    def show_large_composed_view(self):
        """Pop up a large rendering of the full composed level, optionally with the
        Simple A* path overlaid if 'With Simple A*' is checked."""
        scene = self.merge_selected_scenes()
        if not scene:
            messagebox.showinfo("No composed level", "Add at least one image to the composed level first.")
            return

        pil_img = None
        if self.show_astar_var.get():
            pil_img = self._astar_overlay_image(scene)  # None if A* fails/can't produce a path

        if pil_img is None:
            pil_img = self._render_scene_image(scene)

        self._show_image_popup(pil_img, "Composed Level - Large View")

    def _show_image_popup(self, pil_img, title):
        """Show a (possibly large) PIL image in a scrollable popup window."""
        win = tk.Toplevel(self.master)
        win.title(title)
        win.grid_rowconfigure(0, weight=1)
        win.grid_columnconfigure(0, weight=1)

        canvas = tk.Canvas(win, bg="#222222")
        hbar = ttk.Scrollbar(win, orient=tk.HORIZONTAL, command=canvas.xview)
        vbar = ttk.Scrollbar(win, orient=tk.VERTICAL, command=canvas.yview)
        canvas.configure(xscrollcommand=hbar.set, yscrollcommand=vbar.set)
        canvas.grid(row=0, column=0, sticky="nsew")
        vbar.grid(row=0, column=1, sticky="ns")
        hbar.grid(row=1, column=0, sticky="ew")

        photo = ImageTk.PhotoImage(pil_img)
        canvas._photo_ref = photo  # keep a ref so it isn't garbage-collected
        canvas.create_image(0, 0, image=photo, anchor="nw")
        canvas.configure(scrollregion=(0, 0, pil_img.width, pil_img.height))
        win.geometry(f"{min(pil_img.width + 24, 1200)}x{min(pil_img.height + 24, 800)}")

    def get_sample_output(self, idx_or_scene, use_snes_graphics=False):
        if isinstance(idx_or_scene, int):
            if idx_or_scene < len(self.generated_scenes):
                scene = self.generated_scenes[idx_or_scene]
            else:
                tensor = torch.tensor(self.current_levels[idx_or_scene])
                scene = torch.argmax(tensor, dim=0).numpy().tolist()

            if game_selected == "Lode Runner":
                tile_numbers = [[int(num) % len(self.id_to_char) for num in row] for row in scene]
                level = SampleOutput(level=tile_numbers, use_snes_graphics=use_snes_graphics)
            else:
                char_grid = char_grid = scene_to_ascii(scene, self.id_to_char, shorten=False)
                level = SampleOutput(level=char_grid, use_snes_graphics=use_snes_graphics)
            return level
        else:
            # Assume idx_or_scene is a scene (list of lists of tile indices)
            scene = idx_or_scene
            if game_selected == "Lode Runner":
                tile_numbers = [[int(num) % len(self.id_to_char) for num in row] for row in scene]
                level = SampleOutput(level=tile_numbers, use_snes_graphics=use_snes_graphics)
            else:
                char_grid = char_grid = scene_to_ascii(scene, self.id_to_char, shorten=False)
                level = SampleOutput(level=char_grid, use_snes_graphics=use_snes_graphics)
            return level
      
    def play_level(self, idx):
        selected_game = self.game_var.get()
        if selected_game == "Lode Runner":
            level = self.get_sample_output(idx, use_snes_graphics=self.use_snes_graphics.get())
            level.play(game="loderunner", level_idx=1)
        elif selected_game in ("Mega Man (Simple)", "Mega Man (Full)"):
            self._play_megaman_level(idx)
        else:
            level = self.get_sample_output(idx, use_snes_graphics=self.use_snes_graphics.get())
            level.play()

    def edit_level(self, idx):
        scene = self.generated_scenes[idx]
        editor_window = tk.Toplevel(self.master)
        editor_window.title("Level Editor")

        LevelEditor(
            editor_window,
            scene,
            self.id_to_char,
            self.char_to_id,
            self.tile_descriptors,
            self.game_var.get(),
            on_save=lambda updated_scene: self._replace_generated_scene(idx, updated_scene)
        )

    def edit_composed_scene(self, idx, extra_on_save=None):
        """Open the LevelEditor for a scene stored in self.composed_scenes.

        Updates the stored scene and its thumbnail in the bottom strip when saved.
        extra_on_save, if given, is called with the updated scene afterward —
        used by the Mega Man layout editor to refresh its own grid render."""
        scene = self.composed_scenes[idx]
        editor_window = tk.Toplevel(self.master)
        editor_window.title("Level Editor")

        def on_save(updated_scene):
            self.composed_scenes[idx] = updated_scene

            # Refresh the thumbnail shown in the composed-level strip
            rendered = self._render_scene_image(updated_scene)
            thumb = rendered.copy()
            thumb.thumbnail((64, 64))
            photo = ImageTk.PhotoImage(thumb)
            self.composed_thumbnails[idx] = photo
            if idx < len(self.composed_thumbnail_labels):
                self.composed_thumbnail_labels[idx].config(image=photo)
                self.composed_thumbnail_labels[idx].image = photo

            if extra_on_save:
                extra_on_save(updated_scene)

        LevelEditor(
            editor_window,
            scene,
            self.id_to_char,
            self.char_to_id,
            self.tile_descriptors,
            self.game_var.get(),
            on_save=on_save
        )

    def _replace_generated_scene(self, idx, updated_scene):
        self.generated_scenes[idx] = updated_scene 
        self.generated_images[idx] = self._render_scene_image(updated_scene) 
        self._refresh_generated_image(idx)
        self._refresh_generated_caption(idx)

    def _get_current_prompt(self):
        if self.automatic_absence_caption.get():
            return append_absence_captions(self.caption_text.get("1.0", tk.END).strip(), TOPIC_KEYWORDS)
        return self.caption_text.get("1.0", tk.END).strip()

    def _evaluate_scene_caption(self, idx):
        scene = self.generated_scenes[idx]
        prompt = self._get_current_prompt()

        if game_selected == 'Mario':
            actual_caption = assign_caption(scene, self.id_to_char, self.char_to_id, self.tile_descriptors, False, False)
            compare_score, exact_matches, partial_matches, excess_phrases = compare_captions(prompt, actual_caption, return_matches=True, debug=self.debug_caption.get())
        elif game_selected == 'Lode Runner':
            actual_caption = lr_assign_caption(scene, self.id_to_char, self.char_to_id, self.tile_descriptors, False, False)
            compare_score, exact_matches, partial_matches, excess_phrases = lr_compare_captions(prompt, actual_caption, return_matches=True, debug=self.debug_caption.get())
        else:
            actual_caption = mm_assign_caption(scene, self.id_to_char, self.char_to_id, self.tile_descriptors, False, False)
            compare_score, exact_matches, partial_matches, excess_phrases = mm_compare_captions(prompt, actual_caption, return_matches=True, debug=self.debug_caption.get())

        avg_segment_score = None
        if game_selected == "Mario" and len(scene[0]) > common_settings.MARIO_WIDTH:
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
        elif game_selected == "Lode Runner" and len(scene[0]) > common_settings.LR_WIDTH:
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
        elif game_selected not in ["Mario", "Lode Runner"] and len(scene[0]) > common_settings.MEGAMAN_WIDTH:
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

        return exact_matches, partial_matches, excess_phrases, compare_score, avg_segment_score

    def _refresh_generated_caption(self, idx):
        refs = self.generated_widget_refs[idx]
        caption_text_widget = refs.get("caption_text")
        score_label = refs.get("score_label")
        exact_matches, partial_matches, excess_phrases, compare_score, avg_segment_score = self._evaluate_scene_caption(idx)

        caption_text_widget.config(state=tk.NORMAL)
        caption_text_widget.delete(1.0, tk.END)
        for phrase in exact_matches:
            caption_text_widget.insert(tk.END, phrase + ". ", "green")
        for phrase in partial_matches:
            caption_text_widget.insert(tk.END, phrase + ". ", "yellow")
        for phrase in excess_phrases:
            caption_text_widget.insert(tk.END, phrase + ". ", "red")
        caption_text_widget.config(state=tk.DISABLED)

        if avg_segment_score is not None:
            score_label_text = f"Comparison Score: {compare_score}\nAverage Segment Score: {avg_segment_score}"
        else:
            score_label_text = f"Comparison Score: {compare_score}"
        score_label.config(text=score_label_text)

    def _render_scene_image(self, scene): 
        if game_selected == "Lode Runner":
            game_name = "LR"
            num_classes = common_settings.LR_TILE_COUNT
        elif game_selected == "Mega Man (Simple)":
            game_name = "MM-Simple"
            num_classes = common_settings.MM_SIMPLE_TILE_COUNT
        elif game_selected == "Mega Man (Full)":
            game_name = "MM-Full"
            num_classes = common_settings.MM_FULL_TILE_COUNT
        else:
            game_name = "Mario"
            num_classes = common_settings.MARIO_TILE_COUNT

        one_hot = torch.nn.functional.one_hot(
            torch.tensor(scene, dtype=torch.long),
            num_classes=num_classes
        ).float().permute(2, 0, 1).unsqueeze(0)

        pil_img = visualize_samples(one_hot, game=game_name)
        return pil_img[0] if isinstance(pil_img, list) else pil_img

    def _refresh_generated_image(self, idx):
        refs = self.generated_widget_refs[idx]
        refs["astar_overlay_shown"] = False  # showing the plain render again
        pil_img = self.generated_images[idx]
        tk_img = ImageTk.PhotoImage(pil_img)
        refs["image_label"].config(image=tk_img)
        refs["image_label"].image = tk_img

    def _show_astar_overlay_for_index(self, idx):
        """Display the Simple A* path overlay on a single generated image."""
        refs = self.generated_widget_refs[idx]
        overlay = self._astar_overlay_image(self.generated_scenes[idx])
        if overlay is None:
            return
        refs["astar_overlay_shown"] = True
        tk_img = ImageTk.PhotoImage(overlay)
        refs["image_label"].config(image=tk_img)
        refs["image_label"].image = tk_img

    def toggle_all_astar_overlays(self):
        """Called when the 'With Simple A*' checkbox is toggled: show or hide the
        A* path overlay on every currently generated image."""
        show = self.show_astar_var.get()
        for idx in range(len(self.generated_scenes)):
            if show:
                self._show_astar_overlay_for_index(idx)
            else:
                self._refresh_generated_image(idx)

    def _replace_generated_scene(self, idx, updated_scene):
        self.generated_scenes[idx] = updated_scene 
        self.generated_images[idx] = self._render_scene_image(updated_scene) 
        self._refresh_generated_image(idx)
        if self.show_astar_var.get():
            self._show_astar_overlay_for_index(idx)
        self._refresh_generated_caption(idx)

    def _astar_path_for_scene(self, scene, spawn=None, orb=None):
        """Run A* on a single scene and return (pil_image_or_None, solved, stats).

        Shared by the per-image 'Simple A*' overlay and the Mega Man layout A*
        visualizer. spawn/orb are MM-only optional (x, y) cells (the user's placed
        spawn/exit). Raises on import/execution errors so callers can surface them."""
        astar_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "astar")
        if astar_dir not in sys.path:
            sys.path.insert(0, astar_dir)
        from astar_traversability_check import astar_path_image

        game_name = {
            "Mario": "Mario",
            "Lode Runner": "LR",
            "Mega Man (Simple)": "MM-Simple",
            "Mega Man (Full)": "MM-Full",
        }.get(self.game_var.get())
        if game_name is None:
            return None, False, {}
        return astar_path_image(scene, game_name, self.id_to_char, self.tile_descriptors,
                                spawn=spawn, orb=orb)

    def _astar_overlay_image(self, scene):
        """Render scene with its A* path and explored cells.
        Returns a PIL image, or None if the path can't be produced."""
        try:
            img, solved, stats = self._astar_path_for_scene(scene)
        except Exception as e:
            print(f"A* path failed for this scene: {e}")
            return None
        if img is None:
            print("No A* path to draw for this scene.")
            return None
        print(f"A* path: {'traversable' if solved else 'NOT traversable'}  ({stats})")
        return img

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

    def update_mario_only_buttons(self):
        is_mario = self.game_var.get() == "Mario"
        state = tk.NORMAL if is_mario else tk.DISABLED
        self.play_composed_button.config(state=state)
        self.astar_composed_button.config(state=state)
        self.graphics_checkbox.config(state=state)
        self.move_left_button.config(state=state)
        self.move_right_button.config(state=state)
        if not is_mario:
            self.use_snes_graphics.set(False)

        is_megaman = self.game_var.get() in ("Mega Man (Simple)", "Mega Man (Full)")
        self.mm_layout_button.config(state=tk.NORMAL if is_megaman else tk.DISABLED)
        self.save_composed_button.config(state=tk.DISABLED if is_megaman else tk.NORMAL)


    def open_megaman_layout_editor(self):
        global game_selected
        if game_selected not in ("Mega Man (Simple)", "Mega Man (Full)"):
            messagebox.showinfo("Mega Man only", "Switch the game dropdown to a Mega Man mode to use this tool.")
            return
        if not self.composed_scenes:
            messagebox.showinfo(
                "No scenes yet",
                "Use 'Add To Level' on one or more generated images first, then open this tool to arrange them."
            )
            return
        MegaManLayoutEditor(self.master, self)
    

class LevelEditor:
    """
    Grid editor for a single scene.

    Click a grid square to select it (red border). Shift-click to add/remove
    additional squares from the selection. Then click a tile in the side palette
    to apply that tile to every selected square at once.
    Right-click a square: quick cycle-backward on just that square (no selection needed).
    """

    SELECTED_BORDER = "#ff3333"
    UNSELECTED_BORDER = "#999999"
    PALETTE_BORDER = "#cccccc"
    PALETTE_ARMED_BORDER = "#3399ff"
    PALETTE_COLUMNS = 4

    def __init__(self, master, scene, id_to_char, char_to_id, tile_descriptors, game, on_save=None):
        self.master = master
        self.scene = [list(row) for row in scene]
        self.id_to_char = id_to_char
        self.char_to_id = char_to_id
        self.tile_descriptors = tile_descriptors
        self.game = game
        self.on_save = on_save
        

        self.selected_cells = set()

        self.master.title("Level Editor")
        self.master.geometry("700x500")
        self.master.minsize(700, 500)

        outer = ttk.Frame(master, padding=12)
        outer.pack(fill=tk.BOTH, expand=True)

        header = ttk.Label(outer, text="Click a square to select it. Shift-click to select more.",
                            font=("Arial", 12))
        header.pack(anchor="w", pady=(0, 10))

        body = ttk.Frame(outer)
        body.pack(fill=tk.BOTH, expand=True)

        # --- Left: scene grid, in a scroll area so large levels still fit ---
        grid_outer = ttk.Frame(body, borderwidth=1, relief="solid")
        grid_outer.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.grid_canvas = tk.Canvas(grid_outer, highlightthickness=0)
        grid_hbar = ttk.Scrollbar(grid_outer, orient=tk.HORIZONTAL, command=self.grid_canvas.xview)
        grid_vbar = ttk.Scrollbar(grid_outer, orient=tk.VERTICAL, command=self.grid_canvas.yview)
        self.grid_frame = ttk.Frame(self.grid_canvas)
        self.grid_frame.bind("<Configure>", lambda e: self.grid_canvas.configure(scrollregion=self.grid_canvas.bbox("all")))
        self.grid_canvas.create_window((0, 0), window=self.grid_frame, anchor="nw")
        self.grid_canvas.configure(xscrollcommand=grid_hbar.set, yscrollcommand=grid_vbar.set)
        self.grid_canvas.grid(row=0, column=0, sticky="nsew")
        grid_vbar.grid(row=0, column=1, sticky="ns")
        grid_hbar.grid(row=1, column=0, sticky="ew")
        grid_outer.grid_rowconfigure(0, weight=1)
        grid_outer.grid_columnconfigure(0, weight=1)

        # --- Right: palette panel ---
        palette_outer = ttk.Frame(body, width=340, borderwidth=1, relief="solid", padding=12)
        palette_outer.pack(side=tk.LEFT, fill=tk.Y, padx=(12, 0))
        palette_outer.pack_propagate(False)

        ttk.Label(palette_outer, text="Tile Palette", font=("Arial", 13, "bold")).pack(anchor="w")
        ttk.Label(palette_outer, text="Select square(s), then click a tile to apply it.",
                  font=("Arial", 10), foreground="#555555", wraplength=300, justify="left").pack(anchor="w", pady=(2, 10))

        status_row = ttk.Frame(palette_outer)
        status_row.pack(fill=tk.X, pady=(0, 10))
        self.selection_count_label = ttk.Label(status_row, text="0 squares selected", font=("Arial", 10, "italic"))
        self.selection_count_label.pack(side=tk.LEFT)
        ttk.Button(status_row, text="Clear Selection", command=self._clear_selection).pack(side=tk.RIGHT)

        self.hover_info_label = ttk.Label(palette_outer, text=" ", font=("Arial", 10), foreground="#3366aa", wraplength=300)
        self.hover_info_label.pack(anchor="w", pady=(0, 10))

        ttk.Separator(palette_outer, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=(0, 10))

        palette_scroll_outer = ttk.Frame(palette_outer)
        palette_scroll_outer.pack(fill=tk.BOTH, expand=True)
        palette_canvas = tk.Canvas(palette_scroll_outer, highlightthickness=0)
        palette_scrollbar = ttk.Scrollbar(palette_scroll_outer, orient=tk.VERTICAL, command=palette_canvas.yview)
        palette_inner = ttk.Frame(palette_canvas)
        palette_inner.bind("<Configure>", lambda e: palette_canvas.configure(scrollregion=palette_canvas.bbox("all")))
        palette_canvas.create_window((0, 0), window=palette_inner, anchor="nw")
        palette_canvas.configure(yscrollcommand=palette_scrollbar.set)
        palette_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        palette_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.tile_images = self._load_tile_images(game)
        self.tile_photo_images = []
        self.palette_photo_images = []

        self.cell_frames = {}
        self.cell_labels = {}

        for r, row in enumerate(self.scene):
            for c, tile_id in enumerate(row):
                frame = tk.Frame(
                    self.grid_frame,
                    highlightthickness=2,
                    highlightbackground=self.UNSELECTED_BORDER,
                    highlightcolor=self.UNSELECTED_BORDER,
                )
                frame.grid(row=r, column=c, padx=1, pady=1)

                photo = ImageTk.PhotoImage(self.tile_images[tile_id])
                self.tile_photo_images.append(photo)
                label = tk.Label(frame, image=photo, borderwidth=0)
                label.image = photo
                label.pack()

                label.bind("<Button-1>", lambda e, r=r, c=c: self._left_click_cell(r, c, shift=bool(e.state & 0x0001)))
                label.bind("<Button-3>", lambda e, r=r, c=c: self._cycle_cell(r, c, -1))

                self.cell_frames[(r, c)] = frame
                self.cell_labels[(r, c)] = label

        # Palette tiles, arranged in a grid (side-by-side), in cycle order
        self.palette_swatch_frames = {}
        for tile_id in range(len(self.id_to_char)):
            self._add_palette_entry(palette_inner, tile_id)

        controls = ttk.Frame(outer)
        controls.pack(pady=(12, 0))
        ttk.Button(controls, text="Save", command=self.save, width=14).pack(side=tk.LEFT, padx=6)
        ttk.Button(controls, text="Cancel", command=master.destroy, width=14).pack(side=tk.LEFT, padx=6)

    # ------------------------------------------------------------------ selection

    def _left_click_cell(self, row, col, shift):
        cell = (row, col)
        if shift:
            if cell in self.selected_cells:
                self.selected_cells.remove(cell)
            else:
                self.selected_cells.add(cell)
        else:
            self.selected_cells = {cell}
        self._refresh_selection_visuals()

    def _clear_selection(self):
        self.selected_cells = set()
        self._refresh_selection_visuals()

    def _refresh_selection_visuals(self):
        for cell, frame in self.cell_frames.items():
            color = self.SELECTED_BORDER if cell in self.selected_cells else self.UNSELECTED_BORDER
            frame.config(highlightbackground=color, highlightcolor=color)
        n = len(self.selected_cells)
        self.selection_count_label.config(text=f"{n} square{'s' if n != 1 else ''} selected")

    def _cycle_cell(self, row, col, direction):
        current_id = self.scene[row][col]
        next_id = (current_id + direction) % len(self.id_to_char)
        self._paint_cell(row, col, next_id)

    # ------------------------------------------------------------------ palette

    def _tile_hover_text(self, tile_id):
        char = self.id_to_char.get(tile_id, "?")
        descriptor = None
        if self.tile_descriptors:
            descriptor = self.tile_descriptors.get(char)
        if isinstance(descriptor, (list, tuple)) and descriptor:
            return str(descriptor[0])
        elif isinstance(descriptor, str) and descriptor:
            return descriptor
        return f"Tile {tile_id}"

    def _add_palette_entry(self, parent, tile_id):
        col = tile_id % self.PALETTE_COLUMNS
        row = tile_id // self.PALETTE_COLUMNS
        parent.grid_columnconfigure(col, weight=1)

        frame = tk.Frame(
            parent,
            highlightthickness=2,
            highlightbackground=self.PALETTE_BORDER,
            highlightcolor=self.PALETTE_BORDER,
            cursor="hand2",
        )
        frame.grid(row=row, column=col, padx=4, pady=4)

        photo = ImageTk.PhotoImage(self.tile_images[tile_id])
        self.palette_photo_images.append(photo)
        img_label = tk.Label(frame, image=photo, borderwidth=0)
        img_label.image = photo
        img_label.pack(padx=6, pady=6)

        hover_text = self._tile_hover_text(tile_id)
        for widget in (frame, img_label):
            widget.bind("<Button-1>", lambda e, t=tile_id: self._apply_tile_to_selection(t))
            widget.bind("<Enter>", lambda e, text=hover_text: self.hover_info_label.config(text=text))
            widget.bind("<Leave>", lambda e: self.hover_info_label.config(text=" "))

        self.palette_swatch_frames[tile_id] = frame

    def _apply_tile_to_selection(self, tile_id):
        if not self.selected_cells:
            messagebox.showinfo(
                "No squares selected",
                "Click one or more grid squares first (shift-click for multiple), then click a tile here."
            )
            return
        for (row, col) in self.selected_cells:
            self._paint_cell(row, col, tile_id)

    def _paint_cell(self, row, col, tile_id):
        self.scene[row][col] = tile_id
        photo = ImageTk.PhotoImage(self.tile_images[tile_id])
        self.tile_photo_images.append(photo)
        label = self.cell_labels[(row, col)]
        label.config(image=photo)
        label.image = photo

    # ------------------------------------------------------------------ save/cancel

    def save(self):
        self.master.destroy()
        if self.on_save:
            self.on_save(self.scene)

    def cancel(self):
        self.master.destroy()

    def _load_tile_images(self, game):
        if game == "Lode Runner":
            return lr_tiles()
        elif game == "Mega Man (Simple)":
            return mm_tiles("MM-Simple")
        elif game == "Mega Man (Full)":
            return mm_tiles("MM-Full")
        return mario_tiles()

class MegaManLayoutEditor:
    """
    Lets the user arrange the scenes accumulated via 'Add To Level' on a free 2D grid.

    Spawn (Player Start) and exit (Exit Orb) are placed via draggable markers that snap
    to a single tile inside a placed scene. Each marker is one-of-a-kind. On export the
    merged level is first stripped of every stray spawn/exit tile the generator may have
    baked into individual scenes, then exactly the user-placed markers are stamped in -
    so a built level always has at most one spawn and one exit and never inherits the
    invisible leftovers that used to survive clearing or reopening the editor.

    The grid can be zoomed so individual tiles are large enough to target precisely.
    """

    DEFAULT_CELL_PIXELS = 72
    MIN_CELL_PIXELS = 48
    MAX_CELL_PIXELS = 420
    GRID_RADIUS = 8
    MAGNIFIER_SIZE = 260        # pixel size of the popup loupe (square)
    MAGNIFIER_TILE_SPAN = 10     # how many tiles across are shown, centered on the cursor

    # Marker key -> (display label, swatch color, ASCII char stamped into the level).
    # 'P' (player spawn) and 'Z' (exit orb) are the chars the .mmlv converter understands;
    # writing them straight into the ASCII level lets markers work for both the Simple and
    # Full tilesets (the Simple tileset has no spawn/exit tile of its own).
    MARKER_DEFS = {
        "start": ("Player Start", "#FFDD00", "P"),
        "exit":  ("Exit Orb",     "#00FFAA", "Z"),
    }

    # Chars that represent a spawn or exit in any Mega Man ASCII level; stripped on export.
    SPAWN_EXIT_CHARS = ("P", "Z")

    def __init__(self, master, app):
        self.master = master
        self.app = app

        self.placements = {}              # (col, row) -> scene_index
        self.placed_scene_indices = set()
        self.placed_items = {}            # (col, row) -> (image_item_id, text_item_id)

        # marker_placements: key -> (col, row, t_col, t_row)
        # t_col/t_row are *tile* indices within the scene at that cell (snapped, exact).
        self.marker_placements = {}       # "start" / "exit" -> (col, row, t_col, t_row)
        self.marker_canvas_ids = {}       # "start" / "exit" -> (oval_id, text_id)

        # Zoom state and render caches.
        self.cell_pixels = self.DEFAULT_CELL_PIXELS
        self._native_scene_cache = {}     # scene_index -> native PIL render (zoom-independent)
        self._scene_photos = {}           # scene_index -> PhotoImage at current zoom (GC guard)

        self._drag_data = None
        self._drag_window = None

        self._drag_data = None
        self._drag_window = None
        self._magnifier_window = None
        self._magnifier_canvas = None

        self.window = tk.Toplevel(master)
        self.window.title("Mega Man Level Layout")
        self.window.geometry("1050x700")

        # --- Left: palette of unplaced scenes + markers ---
        palette_frame = ttk.Frame(self.window, width=170)
        palette_frame.pack(side=tk.LEFT, fill=tk.Y)
        ttk.Label(palette_frame, text="Unplaced Scenes", font=("Arial", 11, "bold")).pack(pady=(8, 2))
        ttk.Label(palette_frame, text="Drag onto the grid →", wraplength=150).pack(pady=(0, 4))

        # Marker buttons at the top of the palette
        self._marker_palette_frames = {}
        for key, (label, color, _) in self.MARKER_DEFS.items():
            f = tk.Frame(palette_frame, bg=color, bd=2, relief="raised", cursor="fleur")
            f.pack(fill=tk.X, padx=6, pady=3)
            tk.Label(f, text=label, bg=color, font=("Arial", 9, "bold"), fg="#111111").pack(side=tk.LEFT, padx=4, pady=4)
            f.bind("<ButtonPress-1>", lambda e, k=key: self._start_marker_drag(e, k))
            for child in f.winfo_children():
                child.bind("<ButtonPress-1>", lambda e, k=key: self._start_marker_drag(e, k))
            self._marker_palette_frames[key] = f

        ttk.Separator(palette_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=6, pady=6)

        self.palette_canvas = tk.Canvas(palette_frame, width=160, highlightthickness=0)
        palette_scrollbar = ttk.Scrollbar(palette_frame, orient=tk.VERTICAL, command=self.palette_canvas.yview)
        self.palette_inner = ttk.Frame(self.palette_canvas)
        self.palette_inner.bind("<Configure>", lambda e: self.palette_canvas.configure(
            scrollregion=self.palette_canvas.bbox("all")))
        self.palette_canvas.create_window((0, 0), window=self.palette_inner, anchor="nw")
        self.palette_canvas.configure(yscrollcommand=palette_scrollbar.set)
        self.palette_canvas.pack(side=tk.LEFT, fill=tk.Y, expand=True)
        palette_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # --- Right: toolbar + grid canvas ---
        right_frame = ttk.Frame(self.window)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        ttk.Label(
            right_frame,
            text="Drag scenes onto the grid to place them. Right-click a scene to edit or remove it.",
            wraplength=820
        ).pack(side=tk.TOP, fill=tk.X, padx=5, pady=(5, 0))

        toolbar = ttk.Frame(right_frame)
        toolbar.pack(side=tk.TOP, fill=tk.X, pady=5)
        ttk.Button(toolbar, text="Play This Layout",       command=self.play_layout).pack(side=tk.LEFT, padx=5)
        ttk.Button(toolbar, text="Save This Layout As...",  command=self.save_layout).pack(side=tk.LEFT, padx=5)
        ttk.Button(toolbar, text="Show A* Path",            command=self.show_astar_path).pack(side=tk.LEFT, padx=5)
        ttk.Button(toolbar, text="Clear Grid",              command=self.clear_grid).pack(side=tk.LEFT, padx=5)
        ttk.Button(toolbar, text="Open Save Folder",        command=self.open_save_folder).pack(side=tk.LEFT, padx=5)


        # Zoom controls
        ttk.Button(toolbar, text="Zoom -", width=6, command=lambda: self._zoom(1 / 1.25)).pack(side=tk.LEFT, padx=(20, 2))
        self._zoom_label = ttk.Label(toolbar, text="100%", width=6, anchor="center")
        self._zoom_label.pack(side=tk.LEFT)
        ttk.Button(toolbar, text="Zoom +", width=6, command=lambda: self._zoom(1.25)).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="Reset",  width=6, command=self._zoom_reset).pack(side=tk.LEFT, padx=2)

        ttk.Label(toolbar, text="Level Name:").pack(side=tk.LEFT, padx=(20, 5))

        self.level_name_var = tk.StringVar(value="AI_Generated_Level")

        ttk.Entry(
            toolbar,
            textvariable=self.level_name_var,
            width=25
        ).pack(side=tk.LEFT, padx=5)

        canvas_frame = ttk.Frame(right_frame)
        canvas_frame.pack(fill=tk.BOTH, expand=True)
        canvas_frame.grid_rowconfigure(0, weight=1)
        canvas_frame.grid_columnconfigure(0, weight=1)

        self.grid_span = self.GRID_RADIUS * 2 + 1
        self.visible_w, self.visible_h = 820, 560
        self.canvas_size = self.grid_span * self.cell_pixels

        self.grid_canvas = tk.Canvas(canvas_frame, bg="#222222", width=self.visible_w, height=self.visible_h,
                                      scrollregion=(0, 0, self.canvas_size, self.canvas_size))
        hbar = ttk.Scrollbar(canvas_frame, orient=tk.HORIZONTAL, command=self.grid_canvas.xview)
        vbar = ttk.Scrollbar(canvas_frame, orient=tk.VERTICAL,   command=self.grid_canvas.yview)
        self.grid_canvas.configure(xscrollcommand=hbar.set, yscrollcommand=vbar.set)
        self.grid_canvas.grid(row=0, column=0, sticky="nsew")
        vbar.grid(row=0, column=1, sticky="ns")
        hbar.grid(row=1, column=0, sticky="ew")

        self._draw_grid(self.grid_span, self.canvas_size)
        self._center_view()

        # Scroll wheel zooms (centered on the cursor); middle-drag pans.
        self.grid_canvas.bind("<MouseWheel>", self._on_grid_mousewheel)
        self.grid_canvas.bind("<ButtonPress-2>", lambda e: self.grid_canvas.scan_mark(e.x, e.y))
        self.grid_canvas.bind("<B2-Motion>",     lambda e: self.grid_canvas.scan_dragto(e.x, e.y, gain=1))

        self._populate_palette()

    # ------------------------------------------------------------------ grid / zoom

    def _draw_grid(self, grid_span, canvas_size):
        ox, oy = self._cell_to_pixel(0, 0)
        self.grid_canvas.create_rectangle(ox, oy, ox + self.cell_pixels, oy + self.cell_pixels,
                                           fill="#333355", outline="")
        for i in range(grid_span + 1):
            pos = i * self.cell_pixels
            self.grid_canvas.create_line(pos, 0, pos, canvas_size, fill="#444444")
            self.grid_canvas.create_line(0, pos, canvas_size, pos, fill="#444444")

    def _grid_origin_offset(self):
        return self.GRID_RADIUS * self.cell_pixels

    def _cell_to_pixel(self, col, row):
        off = self._grid_origin_offset()
        return off + col * self.cell_pixels, off + row * self.cell_pixels

    def _pixel_to_cell(self, x, y):
        off = self._grid_origin_offset()
        col = int((x - off) // self.cell_pixels)
        row = int((y - off) // self.cell_pixels)
        return col, row

    def _center_view(self):
        half_x = (self.canvas_size / 2 - self.visible_w / 2) / self.canvas_size
        half_y = (self.canvas_size / 2 - self.visible_h / 2) / self.canvas_size
        self.grid_canvas.xview_moveto(max(0, half_x))
        self.grid_canvas.yview_moveto(max(0, half_y))

    def _update_zoom_label(self):
        pct = int(round(100 * self.cell_pixels / self.DEFAULT_CELL_PIXELS))
        self._zoom_label.config(text=f"{pct}%")

    def _zoom(self, factor, anchor=None):
        """Zoom by `factor`, keeping the point under `anchor` (canvas-widget pixel
        coords) fixed on screen. anchor=None keeps the current view center fixed."""
        new = int(round(self.cell_pixels * factor))
        new = max(self.MIN_CELL_PIXELS, min(self.MAX_CELL_PIXELS, new))
        if new == self.cell_pixels:
            return
        if anchor is None:
            anchor = (self.grid_canvas.winfo_width() / 2,
                      self.grid_canvas.winfo_height() / 2)
        ax, ay = anchor
        # World point under the anchor, in the current canvas coords. Everything is laid
        # out linearly from (0, 0), so after zooming it sits at (wx, wy) * ratio.
        wx = self.grid_canvas.canvasx(ax)
        wy = self.grid_canvas.canvasy(ay)
        ratio = new / self.cell_pixels

        self.cell_pixels = new
        self._redraw_all()            # recomputes self.canvas_size
        self._update_zoom_label()

        # Scroll so that scaled world point lands back under the anchor pixel.
        self.grid_canvas.xview_moveto(max(0.0, (wx * ratio - ax) / self.canvas_size))
        self.grid_canvas.yview_moveto(max(0.0, (wy * ratio - ay) / self.canvas_size))

    def _on_grid_mousewheel(self, event):
        """Wheel up = zoom in, wheel down = zoom out, centered on the cursor."""
        self._zoom(1.25 if event.delta > 0 else 1 / 1.25, anchor=(event.x, event.y))
        return "break"   # don't also fire the app-wide mousewheel handler

    def _zoom_reset(self):
        if self.cell_pixels == self.DEFAULT_CELL_PIXELS:
            return
        self.cell_pixels = self.DEFAULT_CELL_PIXELS
        self._redraw_all()
        self._update_zoom_label()
        self._center_view()

    def _redraw_all(self):
        """Rebuild every canvas item from the data model at the current zoom level."""
        self.canvas_size = self.grid_span * self.cell_pixels
        self.grid_canvas.delete("all")
        self.grid_canvas.configure(scrollregion=(0, 0, self.canvas_size, self.canvas_size))
        self._draw_grid(self.grid_span, self.canvas_size)

        self.placed_items.clear()
        self._scene_photos.clear()
        for (col, row), scene_index in self.placements.items():
            self._draw_scene(scene_index, col, row)

        self.marker_canvas_ids.clear()
        for key, (col, row, t_col, t_row) in self.marker_placements.items():
            self._draw_marker(key, col, row, t_col, t_row)

    def _scene_photo(self, scene_index):
        """A PhotoImage of a composed scene rendered to fill the current cell size.
        The native render is cached across zooms; only the resize repeats per zoom."""
        native = self._native_scene_cache.get(scene_index)
        if native is None:
            native = self.app._render_scene_image(self.app.composed_scenes[scene_index])
            self._native_scene_cache[scene_index] = native
        size = self.cell_pixels
        resample = Image.NEAREST if size >= native.width else Image.LANCZOS
        photo = ImageTk.PhotoImage(native.resize((size, size), resample))
        self._scene_photos[scene_index] = photo   # keep a ref so it isn't GC'd
        return photo

    # ------------------------------------------------------------------ palette

    def _populate_palette(self):
        for child in self.palette_inner.winfo_children():
            child.destroy()
        for idx, thumb in enumerate(self.app.composed_thumbnails):
            if idx in self.placed_scene_indices:
                continue
            item_frame = ttk.Frame(self.palette_inner, borderwidth=2, relief="raised")
            item_frame.pack(pady=4, padx=4)
            lbl = tk.Label(item_frame, image=thumb)
            lbl.image = thumb
            lbl.pack()
            ttk.Label(item_frame, text=f"#{idx + 1}").pack()
            lbl.bind("<ButtonPress-1>", lambda e, i=idx: self._start_drag(e, i, None))

    # ------------------------------------------------------------------ scene drag

    def _start_drag(self, event, scene_index, from_cell):
        thumb = self.app.composed_thumbnails[scene_index]
        self._drag_data = {"kind": "scene", "scene_index": scene_index, "from_cell": from_cell}
        self._show_drag_window(event, image=thumb)

    def _show_drag_window(self, event, image=None, text=None, color=None):
        self._drag_window = tk.Toplevel(self.window)
        self._drag_window.overrideredirect(True)
        try:
            self._drag_window.attributes("-topmost", True)
        except Exception:
            pass
        if image:
            lbl = tk.Label(self._drag_window, image=image, bd=0)
            lbl.image = image
            lbl.pack()
        else:
            lbl = tk.Label(self._drag_window, text=text, bg=color, font=("Arial", 9, "bold"),
                           fg="#111111", width=10, height=2)
            lbl.pack()
        self._move_drag_window(event)
        self.window.bind("<Motion>",          self._move_drag_window)
        self.window.bind("<ButtonRelease-1>", self._on_drag_release)

    def _move_drag_window(self, event):
        if self._drag_window:
            self._drag_window.geometry(f"+{event.x_root - 32}+{event.y_root - 32}")
        self._update_magnifier(event)

    def _update_magnifier(self, event):
        """While dragging a marker over a placed scene, show a zoomed loupe near the
        cursor so the exact target tile is easy to see."""
        if not self._drag_data or self._drag_data.get("kind") != "marker":
            self._hide_magnifier()
            return

        cx1 = self.grid_canvas.winfo_rootx()
        cy1 = self.grid_canvas.winfo_rooty()
        cx2 = cx1 + self.grid_canvas.winfo_width()
        cy2 = cy1 + self.grid_canvas.winfo_height()

        if not (cx1 <= event.x_root <= cx2 and cy1 <= event.y_root <= cy2):
            self._hide_magnifier()
            return

        x = self.grid_canvas.canvasx(event.x_root - cx1)
        y = self.grid_canvas.canvasy(event.y_root - cy1)
        col, row = self._pixel_to_cell(x, y)

        if (col, row) not in self.placements:
            self._hide_magnifier()
            return

        scene_index = self.placements[(col, row)]
        scene = self.app.composed_scenes[scene_index]
        scene_h, scene_w = len(scene), len(scene[0])
        t_col, t_row = self._tile_under_pointer(col, row, x, y)

        native = self._native_scene_cache.get(scene_index)
        if native is None:
            native = self.app._render_scene_image(scene)
            self._native_scene_cache[scene_index] = native

        tile_w_native = native.width / scene_w
        tile_h_native = native.height / scene_h

        span = self.MAGNIFIER_TILE_SPAN
        half = span // 2

        c0 = max(0, t_col - half)
        r0 = max(0, t_row - half)
        c1 = min(scene_w, c0 + span)
        r1 = min(scene_h, r0 + span)
        c0 = max(0, c1 - span)
        r0 = max(0, r1 - span)

        left = int(c0 * tile_w_native)
        top = int(r0 * tile_h_native)
        right = int(c1 * tile_w_native)
        bottom = int(r1 * tile_h_native)

        crop = native.crop((left, top, right, bottom))
        zoomed = crop.resize((self.MAGNIFIER_SIZE, self.MAGNIFIER_SIZE), Image.NEAREST)
        photo = ImageTk.PhotoImage(zoomed)

        self._show_magnifier(event, photo)

        # Crosshair over the exact tile the marker would snap to
        tile_px_w = self.MAGNIFIER_SIZE / (c1 - c0)
        tile_px_h = self.MAGNIFIER_SIZE / (r1 - r0)
        hx0 = (t_col - c0) * tile_px_w
        hy0 = (t_row - r0) * tile_px_h
        _, marker_color, _ = self.MARKER_DEFS[self._drag_data["marker_key"]]
        self._magnifier_canvas.delete("highlight")
        self._magnifier_canvas.create_rectangle(
            hx0, hy0, hx0 + tile_px_w, hy0 + tile_px_h,
            outline=marker_color, width=3, tags="highlight"
        )

    def _show_magnifier(self, event, photo):
        if self._magnifier_window is None:
            self._magnifier_window = tk.Toplevel(self.window)
            self._magnifier_window.overrideredirect(True)
            try:
                self._magnifier_window.attributes("-topmost", True)
            except Exception:
                pass
            self._magnifier_canvas = tk.Canvas(
                self._magnifier_window, width=self.MAGNIFIER_SIZE, height=self.MAGNIFIER_SIZE,
                highlightthickness=2, highlightbackground="#ffffff"
            )
            self._magnifier_canvas.pack()

        self._magnifier_canvas.delete("bg")
        self._magnifier_canvas.create_image(0, 0, image=photo, anchor="nw", tags="bg")
        self._magnifier_canvas.image = photo  # keep a ref so it isn't GC'd
        self._magnifier_canvas.tag_lower("bg")  # crosshair stays on top

        # Offset above-right of the cursor so the loupe doesn't sit under your hand
        gx = event.x_root + 30
        gy = event.y_root - self.MAGNIFIER_SIZE - 30
        if gy < 0:
            gy = event.y_root + 30
        self._magnifier_window.geometry(f"+{gx}+{gy}")
        self._magnifier_window.deiconify()

    def _hide_magnifier(self):
        if self._magnifier_window is not None:
            self._magnifier_window.withdraw()

    def _begin_drag_from_cell(self, event, col, row):
        scene_index = self.placements.get((col, row))
        if scene_index is None:
            return
        self._remove_markers_on_cell(col, row)  # markers don't follow a moving scene
        self._clear_cell_visual(col, row)
        del self.placements[(col, row)]
        self.placed_scene_indices.discard(scene_index)
        self._start_drag(event, scene_index, (col, row))

    def _on_drag_release(self, event):
        self.window.unbind("<Motion>")
        self.window.unbind("<ButtonRelease-1>")
        if self._drag_window:
            self._drag_window.destroy()
            self._drag_window = None
        if self._magnifier_window:
            self._magnifier_window.destroy()
            self._magnifier_window = None
            self._magnifier_canvas = None

        drag = self._drag_data
        self._drag_data = None

        if drag is None:
            return

        if drag["kind"] == "marker":
            self._finish_marker_drop(event, drag["marker_key"])
        else:
            self._finish_scene_drop(event, drag)

    def _finish_scene_drop(self, event, drag):
        scene_index = drag["scene_index"]
        from_cell   = drag["from_cell"]

        cx1 = self.grid_canvas.winfo_rootx()
        cy1 = self.grid_canvas.winfo_rooty()
        cx2 = cx1 + self.grid_canvas.winfo_width()
        cy2 = cy1 + self.grid_canvas.winfo_height()

        placed = False
        if cx1 <= event.x_root <= cx2 and cy1 <= event.y_root <= cy2:
            x   = self.grid_canvas.canvasx(event.x_root - cx1)
            y   = self.grid_canvas.canvasy(event.y_root - cy1)
            col, row = self._pixel_to_cell(x, y)
            if (col, row) not in self.placements:
                self._place_scene_at(scene_index, col, row)
                placed = True
            else:
                messagebox.showinfo("Occupied", "That grid cell already has a scene.")

        if not placed and from_cell is not None:
            self._place_scene_at(scene_index, *from_cell)

        self._populate_palette()

    def _place_scene_at(self, scene_index, col, row):
        self.placements[(col, row)] = scene_index
        self.placed_scene_indices.add(scene_index)
        self._draw_scene(scene_index, col, row)

    def _draw_scene(self, scene_index, col, row):
        """Draw the scene image + index label for a cell and wire its mouse bindings."""
        px, py  = self._cell_to_pixel(col, row)
        photo   = self._scene_photo(scene_index)
        img_id  = self.grid_canvas.create_image(px, py, image=photo, anchor="nw")
        text_id = self.grid_canvas.create_text(px + 4, py + 4, text=f"#{scene_index + 1}",
                                                anchor="nw", fill="yellow", font=("Arial", 8))
        self.placed_items[(col, row)] = (img_id, text_id)
        self.grid_canvas.tag_bind(img_id, "<ButtonPress-1>",
                                   lambda e, c=col, r=row: self._begin_drag_from_cell(e, c, r))
        self.grid_canvas.tag_bind(img_id, "<ButtonPress-3>",
                                   lambda e, c=col, r=row: self._show_scene_context_menu(e, c, r))

    def _show_scene_context_menu(self, event, col, row):
        menu = tk.Menu(self.window, tearoff=0)
        menu.add_command(label="Edit Scene", command=lambda: self._edit_scene_at(col, row))
        menu.add_command(label="Remove from Grid", command=lambda: self._remove_from_cell(col, row))
        try:
            menu.tk_popup(event.x_root, event.y_root)
        finally:
            menu.grab_release()

    def _edit_scene_at(self, col, row):
        scene_index = self.placements.get((col, row))
        if scene_index is None:
            return

        def after_save(updated_scene):
            # Invalidate cached renders for this scene and redraw it in place.
            self._native_scene_cache.pop(scene_index, None)
            self._scene_photos.pop(scene_index, None)
            self._clear_cell_visual(col, row)
            self._draw_scene(scene_index, col, row)

        self.app.edit_composed_scene(scene_index, extra_on_save=after_save)

    def _clear_cell_visual(self, col, row):
        items = self.placed_items.pop((col, row), None)
        if items:
            for item_id in items:
                self.grid_canvas.delete(item_id)

    def _remove_markers_on_cell(self, col, row):
        """Drop any spawn/exit marker anchored to this cell (its scene is leaving)."""
        for key, (mc, mr, _tc, _tr) in list(self.marker_placements.items()):
            if (mc, mr) == (col, row):
                self._remove_marker(key)

    def _remove_from_cell(self, col, row):
        self._remove_markers_on_cell(col, row)
        scene_index = self.placements.pop((col, row), None)
        self._clear_cell_visual(col, row)
        if scene_index is not None:
            self.placed_scene_indices.discard(scene_index)
        self._populate_palette()

    def clear_grid(self):
        # Clearing the data model and rebuilding the canvas guarantees no spawn/exit
        # (or any other) visual survives - including orphans from earlier re-placements.
        self.placements.clear()
        self.placed_scene_indices.clear()
        self.marker_placements.clear()
        self._redraw_all()
        self._populate_palette()

    # ------------------------------------------------------------------ marker drag

    def _start_marker_drag(self, event, marker_key):
        label, color, _ = self.MARKER_DEFS[marker_key]
        self._drag_data = {"kind": "marker", "marker_key": marker_key}
        self._show_drag_window(event, text=label, color=color)

    def _finish_marker_drop(self, event, marker_key):
        cx1 = self.grid_canvas.winfo_rootx()
        cy1 = self.grid_canvas.winfo_rooty()
        cx2 = cx1 + self.grid_canvas.winfo_width()
        cy2 = cy1 + self.grid_canvas.winfo_height()

        if not (cx1 <= event.x_root <= cx2 and cy1 <= event.y_root <= cy2):
            return  # dropped outside — do nothing, marker stays in palette

        x = self.grid_canvas.canvasx(event.x_root - cx1)
        y = self.grid_canvas.canvasy(event.y_root - cy1)
        col, row = self._pixel_to_cell(x, y)

        if (col, row) not in self.placements:
            messagebox.showinfo("No scene here",
                                "Drop the marker onto a cell that already has a scene placed in it.")
            return

        t_col, t_row = self._tile_under_pointer(col, row, x, y)
        self._place_marker(marker_key, col, row, t_col, t_row)

    def _tile_under_pointer(self, col, row, x, y):
        """Map a canvas pixel inside a cell to the (t_col, t_row) tile it lands on."""
        scene = self.app.composed_scenes[self.placements[(col, row)]]
        scene_h, scene_w = len(scene), len(scene[0])
        px, py = self._cell_to_pixel(col, row)
        tw = self.cell_pixels / scene_w
        th = self.cell_pixels / scene_h
        t_col = int((x - px) // tw)
        t_row = int((y - py) // th)
        t_col = max(0, min(t_col, scene_w - 1))
        t_row = max(0, min(t_row, scene_h - 1))
        return t_col, t_row

    def _place_marker(self, marker_key, col, row, t_col, t_row):
        # Replace any previous instance of this marker — exactly one spawn, one exit.
        old = self.marker_canvas_ids.pop(marker_key, None)
        if old is not None:
            self.grid_canvas.delete(*old)
        self.marker_placements[marker_key] = (col, row, t_col, t_row)
        self._draw_marker(marker_key, col, row, t_col, t_row)

    def _draw_marker(self, marker_key, col, row, t_col, t_row):
        """Draw a marker centered on its snapped tile, scaled to the current zoom."""
        label, color, _ = self.MARKER_DEFS[marker_key]
        scene = self.app.composed_scenes[self.placements[(col, row)]]
        scene_h, scene_w = len(scene), len(scene[0])
        px, py = self._cell_to_pixel(col, row)
        tw = self.cell_pixels / scene_w
        th = self.cell_pixels / scene_h
        cx = px + (t_col + 0.5) * tw
        cy = py + (t_row + 0.5) * th
        r  = max(5, min(tw, th) * 0.45)

        oval_id = self.grid_canvas.create_oval(
            cx - r, cy - r, cx + r, cy + r,
            fill=color, outline="#000000", width=1
        )
        text_id = self.grid_canvas.create_text(
            cx, cy, text=label[0], fill="#111111", font=("Arial", max(7, int(r)), "bold")
        )
        self.marker_canvas_ids[marker_key] = (oval_id, text_id)

        for cid in (oval_id, text_id):
            self.grid_canvas.tag_bind(
                cid, "<ButtonPress-3>",
                lambda e, k=marker_key: self._remove_marker(k)
            )

    def _remove_marker(self, marker_key):
        ids = self.marker_canvas_ids.pop(marker_key, None)
        if ids:
            for cid in ids:
                self.grid_canvas.delete(cid)
        self.marker_placements.pop(marker_key, None)

    # ------------------------------------------------------------------ build merged scene

    def _marker_grid_positions(self):
        """Absolute (col, row) in the merged grid for each placed marker key.

        Uses the same origin/scene geometry as the merge, so positions line up with
        build_merged_ascii's stamping and build_merged_scene's tile layout."""
        if not self.placements or not self.marker_placements:
            return {}
        scenes = self.app.composed_scenes
        sample = next(iter(self.placements.values()))
        scene_h, scene_w = len(scenes[sample]), len(scenes[sample][0])
        min_col = min(c for c, r in self.placements)
        min_row = min(r for c, r in self.placements)
        return {
            key: ((col - min_col) * scene_w + t_col,
                  (row - min_row) * scene_h + t_row)
            for key, (col, row, t_col, t_row) in self.marker_placements.items()
        }

    def build_merged_scene(self):
        """Merge placed scenes into one tile-ID grid (no spawn/exit handling).
        Used for A* (with explicit or auto spawn/orb) and as the base for export."""
        if not self.placements:
            messagebox.showinfo("Empty layout", "Drag at least one scene onto the grid first.")
            return None

        scenes = self.app.composed_scenes
        dims   = {(len(scenes[i]), len(scenes[i][0])) for i in self.placements.values()}
        if len(dims) > 1:
            messagebox.showerror(
                "Mismatched scene sizes",
                "All scenes must share the same width and height.\n"
                "Found sizes (h×w): " + ", ".join(f"{h}×{w}" for h, w in dims)
            )
            return None
        scene_h, scene_w = next(iter(dims))

        blank_tid = self.app.char_to_id.get("@", 0)

        cols    = [c for c, r in self.placements]
        rows    = [r for c, r in self.placements]
        min_col = min(cols);  max_col = max(cols)
        min_row = min(rows);  max_row = max(rows)

        out_w  = (max_col - min_col + 1) * scene_w
        out_h  = (max_row - min_row + 1) * scene_h
        merged = [[blank_tid for _ in range(out_w)] for _ in range(out_h)]

        for (col, row), scene_index in self.placements.items():
            scene = scenes[scene_index]
            x_off = (col - min_col) * scene_w
            y_off = (row - min_row) * scene_h
            for y, tile_row in enumerate(scene):
                for x, tile in enumerate(tile_row):
                    merged[y_off + y][x_off + x] = tile

        return merged

    def build_merged_ascii(self):
        """The merged level as ASCII rows, ready for the .mmlv converter.

        Every spawn/exit tile baked into the generated scenes is stripped first, then
        exactly the user-placed markers are stamped in. Returns a list of strings,
        or None if there is nothing (or something invalid) to build."""
        merged = self.build_merged_scene()
        if merged is None:
            return None

        grid = [list(r) for r in scene_to_ascii(merged, self.app.id_to_char, shorten=False)]

        # Strip any spawn/exit the generator left inside individual scenes.
        empty = "-" if "-" in self.app.char_to_id else "@"
        for r in range(len(grid)):
            for c in range(len(grid[r])):
                if grid[r][c] in self.SPAWN_EXIT_CHARS:
                    grid[r][c] = empty

        # Stamp the user-placed markers at their absolute tile in the merged grid.
        for marker_key, (out_col, out_row) in self._marker_grid_positions().items():
            _, _, ch = self.MARKER_DEFS[marker_key]
            if 0 <= out_row < len(grid) and 0 <= out_col < len(grid[out_row]):
                grid[out_row][out_col] = ch

        return ["".join(r) for r in grid]

    # ------------------------------------------------------------------ actions

    def play_layout(self):
        merged = self.build_merged_scene()
        if merged is None:
            return

        if "start" not in self.marker_placements:
            if not messagebox.askyesno(
                "No Player Start set",
                "You haven't placed a Player Start marker. The player may not spawn correctly.\n\n"
                "Play anyway?"
            ):
                return

        import os

        try:
            success = self.save_level_files()

            if not success:
                return

            os.startfile("megamaker://")

        except Exception as e:
            messagebox.showerror("Play failed", str(e))

    def show_astar_path(self):
        """Run the simple A* agent on the merged layout and show its path overlaid.

        The user's placed Player Start / Exit Orb markers are used as the A* spawn and
        goal; if either is missing, A* auto-places that one (low-left spawn / right orb)."""
        merged = self.build_merged_scene()
        if merged is None:
            return
        positions = self._marker_grid_positions()
        try:
            img, solved, stats = self.app._astar_path_for_scene(
                merged, spawn=positions.get("start"), orb=positions.get("exit"))
        except Exception as e:
            messagebox.showerror("A* failed", str(e))
            return
        if img is None:
            messagebox.showinfo("A* Path", "No A* path could be drawn for this layout.")
            return
        verdict = "traversable" if solved else "NOT traversable"
        print(f"Composed layout A* path: {verdict}  ({stats})")
        self._show_image_window(img, f"A* Path — {verdict}   {stats}")

    def _show_image_window(self, pil_img, title):
        """Show a (possibly large) PIL image in a scrollable popup window."""
        win = tk.Toplevel(self.window)
        win.title(title)
        win.grid_rowconfigure(0, weight=1)
        win.grid_columnconfigure(0, weight=1)

        canvas = tk.Canvas(win, bg="#222222")
        hbar = ttk.Scrollbar(win, orient=tk.HORIZONTAL, command=canvas.xview)
        vbar = ttk.Scrollbar(win, orient=tk.VERTICAL,   command=canvas.yview)
        canvas.configure(xscrollcommand=hbar.set, yscrollcommand=vbar.set)
        canvas.grid(row=0, column=0, sticky="nsew")
        vbar.grid(row=0, column=1, sticky="ns")
        hbar.grid(row=1, column=0, sticky="ew")

        photo = ImageTk.PhotoImage(pil_img)
        canvas._photo_ref = photo   # keep a ref so it isn't GC'd
        canvas.create_image(0, 0, image=photo, anchor="nw")
        canvas.configure(scrollregion=(0, 0, pil_img.width, pil_img.height))
        win.geometry(f"{min(pil_img.width + 24, 1200)}x{min(pil_img.height + 24, 800)}")

    def save_layout(self):
        success = self.save_level_files()

        if success:
            level_name = self.level_name_var.get().strip()

            if not level_name:
                level_name = "AI_Generated_Level"

            messagebox.showinfo(
                "Saved",
                f"Level saved to:\n{self.levels_dir}\n\n"
                f"Files: {level_name}.txt, {level_name}.mmlv"
            )

    def open_save_folder(self):
        levels_dir = getattr(self, "levels_dir", None)
        if levels_dir is None:
            levels_dir = os.path.join(
                os.path.expanduser("~"),
                "AppData", "Local", "MegaMaker", "Levels"
            )
        os.makedirs(levels_dir, exist_ok=True)
        try:
            os.startfile(levels_dir)
        except Exception as e:
            messagebox.showerror("Couldn't open folder", f"{levels_dir}\n\n{e}")

    def save_level_files(self):
        rows = self.build_merged_ascii()
        if rows is None:
            return False

        levels_dir = os.path.join(
            os.path.expanduser("~"),
            "AppData", "Local", "MegaMaker", "Levels"
        )
        os.makedirs(levels_dir, exist_ok=True)
        self.levels_dir = levels_dir  # remember for "Open Save Folder"

        level_name = self.level_name_var.get().strip() or "AI_Generated_Level"
        txt_path = os.path.join(levels_dir, level_name + ".txt")
        mmlv_path = os.path.join(levels_dir, level_name + ".mmlv")

        try:
            with open(txt_path, 'w') as f:
                f.write("\n".join(rows))

            from megaman.vglc_to_mmlv import convert

            lines = open(txt_path).readlines()
            result = convert(lines, level_name=level_name, author="AI")

            with open(mmlv_path, 'w', encoding='utf-8', newline='\n') as f:
                f.write(result)

            return True

        except Exception as e:
            messagebox.showerror("Save failed", str(e))
            return False

import argparse
def parse_args():
    parser = argparse.ArgumentParser(description="Interactive Tile Level Generator")
    parser.add_argument(
        "--game",
        type=str,
        default="Mario",
        choices=["Mario", "LR", "MM-Simple", "MM-Full"],
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
    elif args.game == "MM-Simple":
        game_selected = "Mega Man (Simple)"
        tileset_path = common_settings.MM_SIMPLE_TILESET
    elif args.game == "MM-Full":
        game_selected = "Mega Man (Full)"
        tileset_path = common_settings.MM_FULL_TILESET

    root = tk.Tk()
    app = CaptionBuilder(root)
    app.load_data(args.load_data)
    app.load_model(args.model_path)

    root.mainloop()