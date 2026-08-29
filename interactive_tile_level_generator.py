import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import subprocess
import os
import json
import re
import torch
import gc
import tokenizer
from datetime import datetime
from PIL import ImageTk
import sys
from util.gui_shared import ParentBuilder, GUI_FONT_SIZE
from level_dataset import visualize_samples, convert_to_level_format, positive_negative_caption_split, mario_tiles, lr_tiles, mm_tiles
from util.sampler import SampleOutput
from captions.caption_match import compare_captions
from captions.LR_caption_match import compare_captions as lr_compare_captions
from captions.MM_caption_match import compare_captions as mm_compare_captions
from create_ascii_captions import assign_caption
from captions.MM2_caption_match import assign_caption as mm2_assign_caption
from captions.MM2_caption_match import compare_captions as mm2_compare_captions
from captions.MM2_caption_match import get_tile_categories, get_char_names
from LR_create_ascii_captions import assign_caption as lr_assign_caption
from MM_create_ascii_captions import assign_caption as mm_assign_caption
from captions.util import extract_tileset
import util.common_settings as common_settings
from util.sampler import scene_to_ascii
from models.pipeline_loader import get_pipeline
from level_dataset import append_absence_captions, remove_duplicate_phrases
from captions.caption_match import TOPIC_KEYWORDS
from models.fdm_pipeline import FDMPipeline
from MegaManLayoutEditor import LevelEditor, MegaManLayoutEditor


# Add the parent directory to sys.path so sibling folders can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

global tileset_path
tileset_path = None  # Global variable for tileset path

# Global constant for GUI font size

GUI_FONT = ("Arial", GUI_FONT_SIZE)


class CaptionBuilder(ParentBuilder):


    global tileset_path
    def __init__(self, master, game, caption_source_keys=None, experiment_log=None):
        global tileset_path
        self.panel_panes = ttk.Panedwindow(master, orient=tk.HORIZONTAL)
        self.panel_panes.pack(fill=tk.BOTH, expand=True)
        super().__init__(self.panel_panes)
        # ParentBuilder stores its container as ``master``; the rest of this GUI
        # needs the actual root for dialogs, popups, and global bindings.
        self.master = master

        self.caption_source_keys = caption_source_keys or []
        self.caption_browser_mode = False
        self.caption_library = []
        self.caption_library_source_scenes = []
        self.filtered_caption_indices = []
        self.selected_training_caption = None
        self.selected_training_scene = None
        self.filtered_captions = []
        self._experiment_log_path = None
        self._experiment_scenes_dir = None
        self._experiment_scene_count = 0
        self._experiment_generation_count = 0
        self._experiment_session_id = None
        self._initialize_experiment_log(experiment_log)

        # Selected game is stored solely in game_var from here on
        initial_game = common_settings.normalize_game_name(game) or "Mario"
        self.game_var = tk.StringVar(value=initial_game)
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
        self.caption_frame = ttk.Frame(self.panel_panes, width=200, borderwidth=2, relief="solid")
        self.panel_panes.insert(0, self.caption_frame)
        
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
        self.width_entry = ttk.Combobox(self.caption_frame, font=GUI_FONT, state="normal")
        self.width_entry.pack()
        self.height_label = ttk.Label(self.caption_frame, text="Height (in tiles):")
        self.height_label.pack()
        self.height_entry = ttk.Combobox(self.caption_frame, font=GUI_FONT, state="normal")
        self.height_entry.pack()

        self.MM_WIDTH_OPTIONS = ["16", "32", "48", "64"]
        self.MM_HEIGHT_OPTIONS = ["16", "32", "48", "64"]   # sent to the model as-is

        self.null_rows_label = ttk.Label(self.caption_frame, text="", style="TLabel")
        self.null_rows_label.pack()

        self.height_entry.bind("<<ComboboxSelected>>", self._update_null_rows_label)

        config = common_settings.get_game_config(self.game_var.get())
        self.width_entry.insert(0, str(config["width"]))
        self.height_entry.insert(0, str(config["height"]))

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
        self.image_frame = ttk.Frame(self.panel_panes, borderwidth=2, relief="solid")
        self.panel_panes.insert(1, self.image_frame)
        
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

        self.mm_layout_button = ttk.Button(row1, text="Build Mega Man Level", command=self.open_megaman_layout_editor, style="TButton")
        self.mm_layout_button.pack(side=tk.LEFT, padx=5)

        self.delete_image_button = ttk.Button(row2, text="Delete Selected Image", command=self.delete_selected_composed_image, style="TButton")
        self.delete_image_button.pack(side=tk.LEFT, padx=10)
        self.clear_composed_button = ttk.Button(row2, text="Clear Composed Level", command=self.clear_composed_level, style="TButton")
        self.clear_composed_button.pack(side=tk.LEFT, padx=10)
        self.save_composed_button = ttk.Button(row2, text="Save Composed Level ASCII", command=self.save_composed_level, style="TButton")
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
        self.game_label = ttk.Label(self.caption_frame, text="Select Game:", style="TLabel")
        self.game_label.pack()
        self.game_dropdown = ttk.Combobox(self.caption_frame, textvariable=self.game_var, values=common_settings.GAME_DISPLAY_NAMES, state="readonly", font=GUI_FONT)
        self.game_dropdown.pack()
        self.game_dropdown.bind("<<ComboboxSelected>>", lambda e: self.update_mario_only_buttons()) 
        self.update_mario_only_buttons() 

        # This catches ordinary button, checkbox, and list interactions without changing
        # the behavior of the individual controls.
        self.master.bind_all("<ButtonRelease-1>", self._log_widget_click, add="+")

    def _initialize_experiment_log(self, participant_id):
        """Set up optional JSONL event logging for a human-study participant."""
        if not participant_id:
            return

        safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(participant_id)).strip("._")
        if not safe_id:
            raise ValueError("--experiment_log must contain at least one letter or number.")

        log_stem = f"experiment_{safe_id}"
        self._experiment_log_path = os.path.abspath(f"{log_stem}.jsonl")
        self._experiment_scenes_dir = os.path.abspath(f"{log_stem}_scenes")
        os.makedirs(self._experiment_scenes_dir, exist_ok=True)
        self._experiment_session_id = datetime.now().strftime("%Y%m%dT%H%M%S%f")
        existing_scene_numbers = []
        for filename in os.listdir(self._experiment_scenes_dir):
            match = re.fullmatch(r"scene_(\d{6})\.json", filename)
            if match:
                existing_scene_numbers.append(int(match.group(1)))
        self._experiment_scene_count = max(existing_scene_numbers, default=0)
        self._log_event(
            "experiment_started",
            participant_id=str(participant_id),
            caption_source_keys=self.caption_source_keys,
            scene_directory=self._experiment_scenes_dir,
        )
        print(f"Experiment logging enabled: {self._experiment_log_path}")

    def _log_event(self, event, **details):
        if not self._experiment_log_path:
            return
        record = {
            "timestamp": datetime.now().isoformat(timespec="milliseconds"),
            "session_id": self._experiment_session_id,
            "event": event,
            **details,
        }
        with open(self._experiment_log_path, "a", encoding="utf-8") as log_file:
            log_file.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _log_widget_click(self, event):
        if not self._experiment_log_path:
            return
        widget = event.widget
        try:
            text = widget.cget("text")
        except tk.TclError:
            text = ""
        self._log_event(
            "widget_clicked",
            widget_class=widget.winfo_class(),
            widget_name=str(widget),
            text=text,
        )
        
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
            tokenizer.OUTPUT_LEVEL = tokenizer.SILENT
            _ = self.pipe(generator=generator, **param_values)
            tokenizer.OUTPUT_LEVEL = tokenizer.WARNING
            # If no exception, enable the checkbox
            self.automatic_absence_caption_checkbox.config(state=tk.NORMAL)
        except Exception as e:
            # If any error, disable the checkbox
            self.automatic_absence_caption_checkbox.config(state=tk.DISABLED)
            self.automatic_absence_caption.set(False)

    def _update_dimension_controls(self, is_megaman):
        if is_megaman:
            self.width_entry.config(values=self.MM_WIDTH_OPTIONS, state="readonly")
            self.height_entry.config(values=self.MM_HEIGHT_OPTIONS, state="readonly")
            if self.width_entry.get() not in self.MM_WIDTH_OPTIONS:
                self.width_entry.set(self.MM_WIDTH_OPTIONS[0])
            if self.height_entry.get() not in self.MM_HEIGHT_OPTIONS:
                self.height_entry.set(self.MM_HEIGHT_OPTIONS[0])
        else:
            self.width_entry.config(values=[], state="normal")
            self.height_entry.config(values=[], state="normal")
        self._update_null_rows_label()

    def _play_megaman_level(self, idx): 
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
        from Game_MMLV.vglc_to_mmlv import convert
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
        char_grid = char_grid = scene_to_ascii(scene, self.id_to_char, shorten=False)


        txt_path = os.path.join(os.getcwd(), "temp_mm_level.txt")
        with open(txt_path, 'w') as f:
            for row in char_grid:
                f.write(''.join(row) + '\n')

        mmlv_path = os.path.join(
            os.path.expanduser("~"),
            "AppData", "Local", "MegaMaker", "Levels", "generated_level.mmlv"
        )
        from Game_MMLV.vglc_to_mmlv import convert
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

    # NOTE: Currently unused. It is tileset aware code that Claude made at one point. 
    def group_phrases(self):
        """Sort the loaded caption phrases into groups that mirror our MM2 tileset.

        MarioDiffusion groups SMB captions with a hardcoded list of substring
        patterns (floor/pipe/cannon/staircase/...). Our captions instead come from
        mm2_tileset_we.json by way of MarioMaker_create_ascii_captions, so we build
        the groups from the tileset itself: every tile is filed into a category by
        its tags, and each phrase lands in the category of the tile it names. The
        style/theme/difficulty metadata and the ground/floor summary, which are not
        tied to any single tile, get their own groups on top.
        """
        global tileset_path

        from Game_MM2.MarioMaker_create_ascii_captions import get_char_names, CAPTION_METADATA_FIELDS

        # Tile char -> lowercase display name, read straight from the tileset tags
        # so the names track exactly what the captioner emits (e.g. "goomba",
        # "question block", "mushroom platform").
        char_names = {char: name.lower() for char, name in get_char_names(tileset_path).items()}

        def tile_category(char):
            """Category for a tile, picked by the first matching tag. Order is only
            match priority: an enemy that is also "damaging"/"hazard" still counts
            as an enemy, and a warp pipe as a pipe rather than a generic block."""
            tags = self.tile_descriptors.get(char, set())
            name = char_names.get(char, "")
            if "enemy" in tags:
                return "Enemies"
            if "collectable" in tags:
                return "Collectables & Power-ups"
            if "hazard" in tags:
                return "Hazards"
            if "platform" in tags:
                return "Platforms"
            if tags & {"pipe", "warp", "door"}:
                return "Pipes, Doors & Warps"
            if "solid" in tags or name.endswith("block"):
                return "Blocks & Terrain"
            return "Other"

        # Match longer names first so "mushroom platform" beats "mushroom" and
        # "bullet bill blaster" beats any shorter overlap.
        named_tiles = sorted(char_names.items(), key=lambda item: len(item[1]), reverse=True)

        # Metadata phrases end in one of these words ("SMW style", "night theme",
        # "easy difficulty"); see MarioMaker_create_ascii_captions. Each suffix
        # gets its own panel rather than one shared "Level Style" bucket.
        metadata_group_names = {
            "style": "Level Style",
            "theme": "Level Theme",
            "difficulty": "Difficulty",
        }
        suffix_to_group = {
            suffix: metadata_group_names.get(suffix, "Level Style")
            for _field, suffix in CAPTION_METADATA_FIELDS
        }

        # Panel order: structural terrain first, enemies/hazards last.
        group_order = [
            "Level Style", "Level Theme", "Difficulty", "Ground & Floor",
            "Blocks & Terrain", "Platforms", "Pipes, Doors & Warps",
            "Collectables & Power-ups", "Enemies", "Hazards", "Other",
        ]
        grouped = {name: [] for name in group_order}

        for phrase in self.all_phrases:
            low = phrase.lower()
            metadata_match = next((suffix for suffix in suffix_to_group if low.endswith(suffix)), None)
            if metadata_match:
                grouped[suffix_to_group[metadata_match]].append(phrase)
            elif "floor" in low or "ground" in low:
                grouped["Ground & Floor"].append(phrase)
            else:
                category = "Other"
                for char, name in named_tiles:
                    if name and name in low:
                        category = tile_category(char)
                        break
                grouped[category].append(phrase)

        # Only show categories that actually occur in the loaded captions.
        return [(name, grouped[name]) for name in group_order if grouped[name]]

    def load_data(self, filepath = None):
        global tileset_path
        if filepath == None:
            filepath = filedialog.askopenfilename(title="Select JSON File", filetypes=[("JSON", "*.json")])
        if filepath:
            _, self.id_to_char, self.char_to_id, self.tile_descriptors = extract_tileset(tileset_path)

            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    dataset = json.load(f)

                if self.caption_source_keys:
                    llm_captions = self._collect_keyed_captions(dataset)
                    if llm_captions:
                        self._show_caption_browser(llm_captions)
                        self._log_event(
                            "caption_dataset_loaded",
                            dataset_path=os.path.abspath(filepath),
                            mode="llm_caption_browser",
                            caption_count=len(llm_captions),
                            caption_source_keys=self.caption_source_keys,
                        )
                        return True

                phrases_set = set()
                for item in dataset:
                    phrases = item['caption'].split('.')
                    phrases_set.update(phrase.strip() for phrase in phrases if phrase.strip())
                    if self.automatic_absence_caption.get():
                        self.update_absence_caption_entry()
                    if self.automatic_negative_caption.get():
                        self.update_negative_prompt_entry
                            
                self.all_phrases = sorted(list(phrases_set))
                self.create_checkboxes()
                self._log_event(
                    "caption_dataset_loaded",
                    dataset_path=os.path.abspath(filepath),
                    mode="structured_phrase_builder",
                    phrase_count=len(self.all_phrases),
                )

                return True
            except FileNotFoundError as e:
                print(f"Error loading data: {e}")
                messagebox.showerror("Error", f"Error loading data: {e}")

        return False

    def _collect_keyed_captions(self, dataset):
        """Return plain-text LLM captions from the requested keyed fields in source order."""
        if not self.caption_source_keys or not isinstance(dataset, list):
            return []

        captions = []
        source_scenes = []
        for item in dataset:
            if not isinstance(item, dict):
                continue
            for key in self.caption_source_keys:
                for caption in self._normalize_caption_values(item.get(key, [])):
                    # Keep the dataset's scene/key/caption order, which is useful
                    # when comparing a prompt to its source data. Keep duplicate
                    # captions too: every list element represents a caption used
                    # for a particular training scene.
                    captions.append(caption)
                    source_scenes.append(item.get("scene"))
        self.caption_library_source_scenes = source_scenes
        return captions

    @staticmethod
    def _normalize_caption_values(value):
        """Flatten JSON-encoded caption lists and remove JSON quoting artifacts.

        Some datasets contain a normal JSON list under a caption key, while others
        contain that same list serialized once more as a string.  The latter used
        to appear as a literal ``[\"caption\"]`` entry in the caption browser.
        """
        if isinstance(value, list):
            captions = []
            for item in value:
                captions.extend(CaptionBuilder._normalize_caption_values(item))
            return captions
        if not isinstance(value, str):
            return []

        text = value.strip()
        if not text:
            return []

        # A trailing comma is a common artifact when a list was copied from a
        # larger JSON object. It is not part of a caption and prevents json.loads.
        json_candidate = text[:-1].rstrip() if text.endswith(",") else text
        if json_candidate.startswith("[") or (
            len(json_candidate) >= 2 and json_candidate[0] == json_candidate[-1] and json_candidate[0] in "\"'"
        ):
            try:
                decoded = json.loads(json_candidate)
            except json.JSONDecodeError:
                decoded = None
            if isinstance(decoded, (list, str)):
                return CaptionBuilder._normalize_caption_values(decoded)

        # Keep a non-JSON string verbatim except for accidental outer quotes.
        if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'":
            text = text[1:-1].strip()
        return [text] if text else []

    def _show_caption_browser(self, captions):
        """Replace the phrase builder with a searchable library of LLM captions."""
        for widget in self.checkbox_frame.winfo_children():
            widget.destroy()

        self.caption_browser_mode = True
        self.caption_library = captions
        self.filtered_caption_indices = list(range(len(captions)))

        controls = ttk.Frame(self.checkbox_frame)
        controls.pack(fill=tk.X, padx=5, pady=5)
        ttk.Button(controls, text="Load Data", command=self.load_data).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(controls, text="Load Model", command=self.load_model).pack(side=tk.LEFT)

        ttk.Label(self.checkbox_frame, text="Training Captions").pack(anchor=tk.W, padx=5)
        ttk.Label(self.checkbox_frame, text="Search captions:").pack(anchor=tk.W, padx=5, pady=(5, 0))
        self.caption_search_var = tk.StringVar()
        self.caption_search_var.trace_add("write", self._filter_caption_library)
        search_entry = ttk.Entry(self.checkbox_frame, textvariable=self.caption_search_var)
        search_entry.pack(fill=tk.X, padx=5)

        list_frame = ttk.Frame(self.checkbox_frame)
        list_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.caption_listbox = tk.Listbox(list_frame, selectmode=tk.BROWSE, activestyle="none", font=GUI_FONT)
        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.caption_listbox.yview)
        self.caption_listbox.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.caption_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.caption_listbox.bind("<<ListboxSelect>>", self._load_selected_library_caption)

        self.caption_library_count_label = ttk.Label(self.checkbox_frame)
        self.caption_library_count_label.pack(anchor=tk.W, padx=5, pady=(0, 5))
        self._refresh_caption_listbox()

    def _filter_caption_library(self, *_):
        query = self.caption_search_var.get().casefold()
        self.filtered_caption_indices = [
            index for index, caption in enumerate(self.caption_library)
            if query in caption.casefold()
        ]
        self._refresh_caption_listbox()
        self._log_event("caption_search_changed", search_text=self.caption_search_var.get(), result_count=len(self.filtered_caption_indices))

    def _refresh_caption_listbox(self):
        self.caption_listbox.delete(0, tk.END)
        for index in self.filtered_caption_indices:
            self.caption_listbox.insert(tk.END, self.caption_library[index])
        self.caption_library_count_label.config(
            text=f"{len(self.filtered_caption_indices)} of {len(self.caption_library)} captions"
        )

    def _load_selected_library_caption(self, _event=None):
        selection = self.caption_listbox.curselection()
        if not selection:
            return
        caption_index = self.filtered_caption_indices[selection[0]]
        caption = self.caption_library[caption_index]
        self.caption_text.delete("1.0", tk.END)
        self.caption_text.insert("1.0", caption)
        self.present_caption = caption
        self.selected_training_caption = caption
        self.selected_training_scene = self.caption_library_source_scenes[caption_index]
        self._log_event("training_caption_selected", caption=caption, has_source_scene=self.selected_training_scene is not None)
        
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
            self._log_event("model_loaded", model_path=os.path.abspath(model), model_name=filename)
    
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
    
    def _selected_game_config(self):
        return common_settings.get_game_config(self.game_var.get())

    def _prepare_scene_output(self, scene, images):
        config = self._selected_game_config()
        if config["is_lode_runner"]:
            actual_caption = lr_assign_caption(scene, self.id_to_char, self.char_to_id, self.tile_descriptors, False, False)
            pil_img = visualize_samples(images, game="LR")
        elif config["is_mario"]:
            actual_caption = assign_caption(scene, self.id_to_char, self.char_to_id, self.tile_descriptors, False, False)
            pil_img = visualize_samples(images)
        elif config["is_mario_maker_2"]:
            _, _, ground_chars = get_tile_categories(tileset_path)
            char_names = get_char_names(tileset_path)
            actual_caption = mm2_assign_caption(scene, self.id_to_char, char_names, ground_chars)
            pil_img = visualize_samples(images, game="MM2")
        else:
            actual_caption = mm_assign_caption(scene, self.id_to_char, self.char_to_id, self.tile_descriptors, False, False)
            pil_img = visualize_samples(images, game=config["render_name"])
        return actual_caption, pil_img

    def _compare_caption(self, prompt, actual_caption):
        config = self._selected_game_config()
        if config["is_lode_runner"]:
            return lr_compare_captions(prompt, actual_caption, return_matches=True, debug=self.debug_caption.get())
        if config["is_mario"]:
            return compare_captions(prompt, actual_caption, return_matches=True, debug=self.debug_caption.get())
        if config["is_mario_maker_2"]:
            return mm2_compare_captions(prompt, actual_caption, return_matches=True, debug=self.debug_caption.get())
        return mm_compare_captions(prompt, actual_caption, return_matches=True, debug=self.debug_caption.get())

    def generate_image(self):
        global tileset_path
        # # cannot use multiple generations of levels in one composed level
        # self.clear_composed_level()
        # print("Clearing previously composed level for newly generated scenes.")

        # clear the previous images
        self.generated_images = []
        self.generated_scenes = []

        self.generated_widget_refs = [] 

        print("Generating")
        
        prompt = self._get_current_prompt()
        original_scene = None
        if prompt == self.selected_training_caption and isinstance(self.selected_training_scene, list):
            original_scene = self.selected_training_scene
        
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

        self._experiment_generation_count += 1
        generation_id = self._experiment_generation_count
        self._log_event(
            "generation_requested",
            generation_id=generation_id,
            caption=prompt,
            negative_prompt=negative_prompt,
            requested_image_count=num_images,
            seed=int(self.seed_entry.get()),
            parameters=param_values,
            uses_unmodified_training_caption=original_scene is not None,
        )

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

                config = self._selected_game_config()

                chop_rows = 0
                if config["is_megaman"]:
                    try:
                        chop_rows = (int(self.height_entry.get()) // 16) * 2
                    except ValueError:
                        chop_rows = 0
                if chop_rows > 0:
                    images = images[:, :, chop_rows:, :]

                self.current_levels.append(images[0].cpu().detach().numpy())

                sample_tensor = images[0].unsqueeze(0)
                sample_indices = convert_to_level_format(sample_tensor)
                scene = sample_indices[0].tolist()

                print(f"Update tileset for game: {self.game_var.get()}")
                scene = [[x % config["tile_count"] for x in row] for row in scene]
                tileset_path = config["tileset"]
                _, self.id_to_char, self.char_to_id, self.tile_descriptors = extract_tileset(tileset_path)

                self.generated_scenes.append(scene)
                print(f"Assigning caption for game: {self.game_var.get()}")
                actual_caption, pil_img = self._prepare_scene_output(scene, images)

                self._save_experiment_scene(
                    scene=scene,
                    image=pil_img,
                    generation_id=generation_id,
                    image_index=i,
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    parameters=param_values,
                )

                self.generated_images.append(pil_img)
                img_tk = ImageTk.PhotoImage(pil_img)
                print(f"Comparing captions for game: {self.game_var.get()}")
                compare_score, exact_matches, partial_matches, excess_phrases = self._compare_caption(prompt, actual_caption)

            except Exception as e:
                self._log_event(
                    "generation_failed",
                    generation_id=generation_id,
                    image_index=i,
                    error=str(e),
                )
                messagebox.showerror(
                    "Generation Error",
                    f"Failed to generate image {i + 1}.\n\n"
                    f"This may be caused by selecting the wrong game for the loaded model.\n\n"
                    f"Details: {str(e)}"
                )
                break

            row_frame = ttk.Frame(self.image_inner_frame)
            row_frame.grid(row=i, column=0, pady=10, sticky="n")
            if original_scene is not None:
                original_frame = ttk.Frame(row_frame)
                original_frame.grid(row=0, column=0, padx=(0, 12), sticky="n")
                img_frame = ttk.Frame(row_frame)
                img_frame.grid(row=0, column=1, sticky="n")
            else:
                original_frame = None
                img_frame = ttk.Frame(row_frame)
                img_frame.grid(row=0, column=0, sticky="n")


            print(f"Image {i + 1} dimensions: width={img_tk.width()}, height={img_tk.height()}")

            # When comparing with a training scene, reserve half the row for each image.
            display_width = max(1, frame_width // 2) if original_frame else frame_width
            if img_tk.width() > display_width:
                scale_factor = display_width / img_tk.width()
                new_size = (display_width, max(1, int(img_tk.height() * scale_factor)))
                img_tk = ImageTk.PhotoImage(pil_img.resize(new_size))
                print(f"Image {i + 1} scaled to: width={new_size[0]}, height={new_size[1]}")

            if original_frame is not None:
                try:
                    original_pil_img = self._render_scene_image(original_scene)
                    original_tk = ImageTk.PhotoImage(original_pil_img)
                    if original_tk.width() > display_width:
                        original_scale = display_width / original_tk.width()
                        original_size = (display_width, max(1, int(original_tk.height() * original_scale)))
                        original_tk = ImageTk.PhotoImage(original_pil_img.resize(original_size))
                    original_label = ttk.Label(original_frame, image=original_tk)
                    original_label.image = original_tk
                    original_label.pack()
                    ttk.Label(original_frame, text="Original").pack(pady=(5, 0))
                except Exception as error:
                    # Generation remains usable even if a malformed source scene cannot be rendered.
                    print(f"Could not render original training scene: {error}")

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
            if self._selected_game_config()["is_mario"]:
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

            self.generated_widget_refs.append({
                "image_label": label,
                "caption_text": caption_text,
                "score_label": score_label,
            })

            # Create a frame for buttons
            button_frame = ttk.Frame(img_frame)
            button_frame.pack(pady=5)
    
            config = self._selected_game_config()
            if config["supports_per_image_play"]:
                # Add Play button
                play_button = ttk.Button(
                    button_frame,
                    text="Play",
                    command=lambda idx=i: self.play_level(idx),
                    style="TButton"
                )
                play_button.pack(side=tk.LEFT, padx=5)

                if config["is_mario"]:
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
        self._log_event("generation_completed", generation_id=generation_id, generated_image_count=len(self.generated_scenes))
        #print(self.current_levels)

    def _save_experiment_scene(self, scene, image, generation_id, image_index, prompt, negative_prompt, parameters):
        """Persist each generated scene and its generation context for study analysis."""
        if not self._experiment_scenes_dir:
            return

        self._experiment_scene_count += 1
        scene_stem = f"scene_{self._experiment_scene_count:06d}"
        scene_path = os.path.join(self._experiment_scenes_dir, f"{scene_stem}.json")
        image_path = os.path.join(self._experiment_scenes_dir, f"{scene_stem}.png")
        record = {
            "scene_id": scene_stem,
            "session_id": self._experiment_session_id,
            "generation_id": generation_id,
            "image_index": image_index,
            "timestamp": datetime.now().isoformat(timespec="milliseconds"),
            "caption": prompt,
            "negative_prompt": negative_prompt,
            "parameters": parameters,
            "scene": scene,
        }
        with open(scene_path, "w", encoding="utf-8") as scene_file:
            json.dump(record, scene_file, ensure_ascii=False, indent=2)
        image.save(image_path)
        self._log_event(
            "scene_generated",
            scene_id=scene_stem,
            generation_id=generation_id,
            image_index=image_index,
            scene_path=scene_path,
            image_path=image_path,
            caption=prompt,
        )

    def add_to_composed_level(self, idx):
        # Assigns tileset_path below, so it must be declared global 
        global tileset_path
        # Store the actual scene
        scene = self.generated_scenes[idx]
        if self.game_var.get() == "Lode Runner":
                number_of_tiles = common_settings.LR_TILE_COUNT
                scene = [[x % number_of_tiles for x in row] for row in scene]
                tileset_path = common_settings.LR_TILESET
        elif self.game_var.get() == "Mega Man (Simple)":
                number_of_tiles = common_settings.MM_SIMPLE_TILE_COUNT
                scene = [[x % number_of_tiles for x in row] for row in scene]
                tileset_path = common_settings.MM_SIMPLE_TILESET
        elif self.game_var.get() == "Mega Man (Full)":
            number_of_tiles = common_settings.MM_FULL_TILE_COUNT
            scene = [[x % number_of_tiles for x in row] for row in scene]
            tileset_path = common_settings.MM_FULL_TILESET
        elif self.game_var.get() == "Mega Man (Maker)":
            number_of_tiles = common_settings.MMLV_TILE_COUNT
            scene = [[x % number_of_tiles for x in row] for row in scene]
            tileset_path = common_settings.MMLV_TILESET
        # Mario and Mario Maker need no case here as the Tile Id's are already in range and correct
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
            # Mario Maker exports a .swe into SMM:WE's level folder and launches the game
            if self.game_var.get() == "Mario Maker 2":
                self._play_composed_swe()
                return
            level = self.get_sample_output(scene, use_snes_graphics=self.use_snes_graphics.get())
            level.play()

    def save_composed_level(self):
        scene = self.merge_selected_scenes()
        if scene:
            # Mario Maker saves a .swe into SMM:WE's level folder instead of a .txt
            # To save an SWE, choose to play the level. This option saves ASCII instead
            #if self.game_var.get() == "Mario Maker 2":
            #    self._save_composed_swe()
            #    return

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

    def _smmwe_niveles_dir(self):
        """SMM:WE's level folder, %LOCALAPPDATA%\\SMM_WE\\Niveles. Falls back to a
        local folder when LOCALAPPDATA isn't set (non-Windows)."""
        base = os.environ.get("LOCALAPPDATA")
        if base:
            return os.path.join(base, "SMM_WE", "Niveles")
        return os.path.join(os.getcwd(), "Niveles")

    def _smmwe_exe_search_paths(self):
        """Return candidate paths where SMM:WE may be installed."""
        paths = []
        for env in ("ProgramFiles(x86)", "ProgramFiles", "ProgramW6432"):
            base = os.environ.get(env)
            if base:
                paths.append(os.path.join(base, "SMMWE", "SMM_WE.exe"))
        paths.extend([
            r"C:\Program Files (x86)\SMMWE\SMM_WE.exe",
            r"C:\Program Files\SMMWE\SMM_WE.exe",
        ])
        return paths

    def _smmwe_exe_path(self):
        """Path to SMM_WE.exe (installs to Program Files\\SMMWE), or None."""
        for exe in self._smmwe_exe_search_paths():
            if os.path.isfile(exe):
                return exe
        return None

    def _compose_swe_bytes(self, name):
        """Convert the merged composed scene to a .swe (ascii -> json -> swe).
        Returns (swe_bytes, dropped_counts)."""
        from mm2pipeline_data.ascii import ascii_to_level
        from mm2pipeline_data.swe import build_world, encode_swe, detect_smmwe_user
        from datetime import datetime

        sample = self.get_sample_output(self.merge_selected_scenes())
        # '_' is padding, not a real tile, but the converter reads it as Goal
        # Ground. Treat it as empty space so it doesn't litter the level.
        ascii_text = "\n".join(row.replace("_", " ") for row in sample.level)

        level_json = ascii_to_level(ascii_text, source_file=name)

        now = datetime.now()
        s0, dropped = build_world(
            level_json,
            user=detect_smmwe_user(),
            name=name,
            desc=None,
            date_str=now.strftime("%d/%m/%Y"),
            time_str=now.strftime("%H:%M"),
        )
        return encode_swe({"S0": s0, "SB1": {"S1": []}}), dropped

    @staticmethod
    def _report_dropped(dropped):
        if dropped:
            total = sum(dropped.values())
            summary = ", ".join(f"{n}x {nm}" for nm, n in
                                sorted(dropped.items(), key=lambda kv: -kv[1]))
            print(f"  dropped {total} object(s) with no SMM:WE equivalent: {summary}")

    def _save_composed_swe(self):
        """Save the composed scene as a .swe, prompting for a name in Niveles."""
        exe = self._smmwe_exe_path()
        if exe is None:
            search_paths = self._smmwe_exe_search_paths()
            search_text = "\n".join(search_paths)
            message = (
                "Could not find the SMM:WE executable.\n"
                "SMM_WE.exe was searched for in the following locations:\n\n"
                f"{search_text}\n\n"
                "Please install SMM:WE or place SMM_WE.exe in one of these folders."
            )
            messagebox.showerror("SMM:WE executable not found", message)
            print("SMM:WE executable not found. Looked in the following locations:")
            for path in search_paths:
                print(f"  {path}")
            return

        niveles_dir = self._smmwe_niveles_dir()
        os.makedirs(niveles_dir, exist_ok=True)
        file_path = filedialog.asksaveasfilename(
            defaultextension=".swe",
            filetypes=[("SMM:WE level", "*.swe")],
            title="Save Composed Level to SMM:WE",
            initialdir=niveles_dir,
            initialfile="composed_level.swe",
        )
        if not file_path:
            print("Save operation cancelled.")
            return

        name = os.path.splitext(os.path.basename(file_path))[0]
        swe_bytes, dropped = self._compose_swe_bytes(name)
        with open(file_path, "wb") as f:
            f.write(swe_bytes)
        print(f"Composed level exported to {file_path} ({len(swe_bytes)} bytes)")
        self._report_dropped(dropped)

    def _play_composed_swe(self):
        """Save the composed level to Niveles and launch SMM:WE. There's no way
        to boot straight into a level, so you pick 'composed_level' in-game."""
        import subprocess

        exe = self._smmwe_exe_path()
        if exe is None:
            search_paths = self._smmwe_exe_search_paths()
            search_text = "\n".join(search_paths)
            message = (
                "Could not find the SMM:WE executable.\n"
                "SMM_WE.exe was searched for in the following locations:\n\n"
                f"{search_text}\n\n"
                "Please install SMM:WE or place SMM_WE.exe in one of these folders."
            )
            messagebox.showerror("SMM:WE executable not found", message)
            print("SMM:WE executable not found. Looked in the following locations:")
            for path in search_paths:
                print(f"  {path}")
            return

        name = "composed_level"
        niveles_dir = self._smmwe_niveles_dir()
        os.makedirs(niveles_dir, exist_ok=True)
        swe_bytes, dropped = self._compose_swe_bytes(name)
        out_path = os.path.join(niveles_dir, name + ".swe")
        with open(out_path, "wb") as f:
            f.write(swe_bytes)
        print(f"Composed level exported to {out_path} ({len(swe_bytes)} bytes)")
        self._report_dropped(dropped)
        # run from the install dir so the game finds data.win
        subprocess.Popen([exe], cwd=os.path.dirname(exe))
        print(f"Launched SMM:WE -- open the level browser and play '{name}'.")

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

            if self.game_var.get() == "Lode Runner":
                tile_numbers = [[int(num) % len(self.id_to_char) for num in row] for row in scene]
                level = SampleOutput(level=tile_numbers, use_snes_graphics=use_snes_graphics)
            elif self.game_var.get() == "Mario Maker 2":
                # Shorten needs to be false for Mario Maker otherwise the top rows would be axed. It has different logic.
                char_grid = scene_to_ascii(scene, self.id_to_char, shorten=False)
                level = SampleOutput(level=char_grid, use_snes_graphics=use_snes_graphics)
            else:
                char_grid = char_grid = scene_to_ascii(scene, self.id_to_char, shorten=False)
                level = SampleOutput(level=char_grid, use_snes_graphics=use_snes_graphics)
            return level
        else:
            # Assume idx_or_scene is a scene (list of lists of tile indices)
            scene = idx_or_scene
            if self.game_var.get() == "Lode Runner":
                tile_numbers = [[int(num) % len(self.id_to_char) for num in row] for row in scene]
                level = SampleOutput(level=tile_numbers, use_snes_graphics=use_snes_graphics)
            elif self.game_var.get() == "Mario Maker 2":
                # Shorten needs to be false for Mario Maker otherwise the top rows would be axed. It has different logic.
                char_grid = scene_to_ascii(scene, self.id_to_char, shorten=False)
                level = SampleOutput(level=char_grid, use_snes_graphics=use_snes_graphics)
            else:
                char_grid = char_grid = scene_to_ascii(scene, self.id_to_char, shorten=False)
                level = SampleOutput(level=char_grid, use_snes_graphics=use_snes_graphics)

            return level
      
    def play_level(self, idx):
        selected_game = self.game_var.get()
        if selected_game == "Lode Runner":
            level = self.get_sample_output(idx, use_snes_graphics=self.use_snes_graphics.get())
            level.play(game="loderunner", level_idx=1)
        elif selected_game in ("Mega Man (Simple)", "Mega Man (Full)", "Mega Man (Maker)"):
            self._play_megaman_level(idx)
        else:
            # Mario. (Per-image play is only enabled for Mario; Mario Maker plays
            # via the composed-level flow, which exports a .swe and launches SMM:WE.)
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

        if self.game_var.get() == 'Mario':
            actual_caption = assign_caption(scene, self.id_to_char, self.char_to_id, self.tile_descriptors, False, False)
            compare_score, exact_matches, partial_matches, excess_phrases = compare_captions(prompt, actual_caption, return_matches=True, debug=self.debug_caption.get())
        elif self.game_var.get() == 'Lode Runner':
            actual_caption = lr_assign_caption(scene, self.id_to_char, self.char_to_id, self.tile_descriptors, False, False)
            compare_score, exact_matches, partial_matches, excess_phrases = lr_compare_captions(prompt, actual_caption, return_matches=True, debug=self.debug_caption.get())
        elif self.game_var.get() == 'Mario Maker 2':
            actual_caption = self.mm2_assign_caption(scene)
            compare_score, exact_matches, partial_matches, excess_phrases = self.mm_compare_captions(prompt, actual_caption, return_matches=True, debug=self.debug_caption.get())
        else:  # Mega Man variants
            actual_caption = mm_assign_caption(scene, self.id_to_char, self.char_to_id, self.tile_descriptors, False, False)
            compare_score, exact_matches, partial_matches, excess_phrases = mm_compare_captions(prompt, actual_caption, return_matches=True, debug=self.debug_caption.get())

        avg_segment_score = None
        if self.game_var.get() == "Mario" and len(scene[0]) > common_settings.MARIO_WIDTH:
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
        config = self._selected_game_config()
        game_name = config["render_name"]
        num_classes = config["tile_count"]

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

        game_name = self._selected_game_config()["render_name"]
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
        config = self._selected_game_config()
        is_mario = config["is_mario"]
        state = tk.NORMAL if is_mario else tk.DISABLED
        self.astar_composed_button.config(state=state)
        self.graphics_checkbox.config(state=state)
        self.move_left_button.config(state=state)
        self.move_right_button.config(state=state)
        if not is_mario:
            self.use_snes_graphics.set(False)

        is_playable = config["is_composed_playable"]
        state = tk.NORMAL if is_playable else tk.DISABLED
        self.play_composed_button.config(state=state)

        is_megaman = config["is_megaman"]
        self.mm_layout_button.config(state=tk.NORMAL if is_megaman else tk.DISABLED)
        self.save_composed_button.config(state=tk.DISABLED if is_megaman else tk.NORMAL)
        self._update_dimension_controls(is_megaman)

    def _update_null_rows_label(self, event=None):
        is_megaman = self._selected_game_config()["is_megaman"]
        if not is_megaman:
            self.null_rows_label.config(text="")
            return
        try:
            height = int(self.height_entry.get())
        except ValueError:
            self.null_rows_label.config(text="")
            return
        chop = (height // 16) * 2
        self.null_rows_label.config(text=f"({chop} null row{'s' if chop != 1 else ''} chopped from top)")

    def open_megaman_layout_editor(self):
        if not self._selected_game_config()["is_megaman"]:
            messagebox.showinfo("Mega Man only", "Switch the game dropdown to a Mega Man mode to use this tool.")
            return
        if not self.composed_scenes:
            messagebox.showinfo(
                "No scenes yet",
                "Use 'Add To Level' on one or more generated images first, then open this tool to arrange them."
            )
            return
        MegaManLayoutEditor(self.master, self)
    
import argparse
def parse_args():
    parser = argparse.ArgumentParser(description="Interactive Tile Level Generator")
    parser.add_argument(
        "--game",
        type=str,
        default="Mario",
        choices=common_settings.GAME_CLI_CHOICES,
        help="Which game to create a model for (affects sample style and tile count)"
    )
    parser.add_argument("--model_path", type=str, help="Path to the trained diffusion model")
    parser.add_argument("--load_data", type=str, default="Game_Mario/DATA/Mar1and2_LevelsAndCaptions-regular.json", help="Path to the dataset JSON file")
    parser.add_argument("--tileset", default=common_settings.MARIO_TILESET, help="Descriptions of individual tile types")
    parser.add_argument(
        "--caption_source_keys",
        nargs="+",
        default=None,
        help="LLM caption-list keys in --load_data. Enables the searchable training-caption browser."
    )
    parser.add_argument(
        "--experiment_log",
        type=str,
        default=None,
        metavar="PARTICIPANT_ID",
        help="Participant ID for JSONL interaction logging and per-scene experiment output."
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    config = common_settings.get_game_config(args.game)
    game = config["name"]
    tileset_path = config["tileset"]

    root = tk.Tk()
    app = CaptionBuilder(
        root,
        game,
        caption_source_keys=args.caption_source_keys,
        experiment_log=args.experiment_log,
    )
    app.load_data(args.load_data)
    app.load_model(args.model_path)

    root.mainloop()
