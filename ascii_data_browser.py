"""
Tile Dataset Viewer
====================

A Tkinter GUI for browsing tile-based level datasets (Mario, Lode Runner,
Mega Man variants, Mario Maker 2), inspecting/regenerating their captions,
and composing several scenes into a single playable/exportable level.

The file is organized into the following sections (search for the "# ----"
banners to jump between them):

    1. Imports & constants
    2. App bootstrap (__init__)
    3. UI construction (create_widgets and friends)
    4. File loading (dataset / tileset)
    5. Rendering (the main canvas: grid view, image view, real-image view, A*)
    6. Attribute / caption panel
    7. Sample navigation
    8. View-mode toggles (grid/image, A* overlay, filter reason, MMLV link)
    9. Composed level management (add/move/delete/merge/edit thumbnails)
    10. Playback & export (Java sim playback, A* check, SMM:WE .swe export)
    11. App lifecycle
    12. Command-line entry point
"""

# ---------------------------------------------------------------------------
# 1. Imports & constants
# ---------------------------------------------------------------------------
import colorsys
import json
import os
import random
import sys
import webbrowser

import torch
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image
import PIL.ImageTk

import level_dataset
import util.common_settings as common_settings
from util.sampler import SampleOutput, scene_to_ascii
from captions.MM2_caption_match import assign_caption as mm2_assign_caption
from captions.MM2_caption_match import get_char_names, get_tile_categories
from captions.util import extract_tileset
from create_ascii_captions import assign_caption
from LR_create_ascii_captions import assign_caption as lr_assign_caption
from MegaManLayoutEditor import LevelEditor, MegaManLayoutEditor
from MM_create_ascii_captions import assign_caption as mm_assign_caption

# Tag -> RGB color used to shade tiles in the MM2 grid view when no fixed
# per-game palette entry applies (see _build_color_map).
MM2_TAG_COLORS = [
    ("empty",       (0.20, 0.30, 0.70)),
    ("air",         (0.20, 0.30, 0.70)),
    ("pipe",        (0.00, 0.55, 0.10)),
    ("warp",        (0.10, 0.70, 0.20)),
    ("door",        (0.10, 0.70, 0.20)),
    ("goal",        (0.00, 0.90, 0.30)),
    ("spawn",       (0.00, 0.00, 0.90)),
    ("enemy",       (0.90, 0.10, 0.10)),
    ("damaging",    (0.90, 0.10, 0.10)),
    ("hazard",      (1.00, 0.50, 0.00)),
    ("collectable", (1.00, 0.85, 0.00)),
    ("item",        (1.00, 0.85, 0.00)),
    ("platform",    (0.30, 0.50, 0.90)),
    ("passable",    (0.70, 0.85, 1.00)),
    ("solid",       (0.50, 0.35, 0.10)),
    ("decoration",  (0.60, 0.60, 0.60)),
]

# Fraction of screen height/width the app is allowed to occupy on launch, so
# the window (and everything packed inside it) never starts taller/wider
# than the screen -- see _fit_window_to_screen.
MAX_SCREEN_FRACTION = 0.90


class TileViewer(tk.Tk):
    # keys we don't want in the attribute dropdown (handled elsewhere)
    RESERVED_ATTRS = {"scene", "details", "data"}

    # -----------------------------------------------------------------
    # 2. App bootstrap
    # -----------------------------------------------------------------
    def __init__(self, dataset_path=None, game=None):
        super().__init__()
        self.title("Tile Dataset Viewer")

        # --- state -----------------------------------------------------
        self.dataset = []
        self.id_to_char = {}
        self.current_sample_idx = 0
        self.current_caption_idx = 0  # which value of the picked attribute is shown
        self.show_prompt = False      # text box shows 'prompt' instead of the attribute
        self.show_ids = tk.BooleanVar(value=False)
        self.describe_absence = tk.BooleanVar(value=False)
        self.show_images = False        # image view vs numeric/character grid
        self.show_astar_path = False    # overlay the A* path on the image view
        self.show_filter_reason = False # show the entry's 'filter_reason' field
        self.added_sample_indexes = []
        self.composed_thumbnails = []
        self.selected_thumb_idx = None
        self.current_pil_image = None   # last-rendered PIL image, for "Save Image As"

        # --- sizing: everything below is derived from screen size so the
        # window (and the grid inside it) scales sensibly on any monitor,
        # and _fit_window_to_screen (called at the end of __init__) makes
        # sure the whole window still fits once every widget is packed. ---
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        self.window_size = min(screen_width, screen_height) * 0.65
        self.tile_size = int(self.window_size / 20)
        self.font_size = max(self.tile_size // 4, 6)

        # --- build the UI (scrollable, so low-resolution screens can still
        # reach every control -- see create_widgets) ---
        self.create_widgets(game)
        self.bind_keys()

        # --- load initial dataset/tileset, if given on the command line ---
        self.dataset_path = dataset_path
        config = common_settings.get_game_config(game)
        self.tileset_path = config["tileset"]
        if self.dataset_path and self.tileset_path:
            self.load_files_from_paths(self.dataset_path, self.tileset_path)

        # --- right-click "Save Image As..." on the main canvas ---
        self.canvas_context_menu = tk.Menu(self, tearoff=0)
        self.canvas_context_menu.add_command(label="Save Image As...", command=self.save_current_image_as)
        self.canvas.bind("<Button-3>", self.show_canvas_context_menu)
        self.canvas.bind("<Control-Button-1>", self.show_canvas_context_menu)  # macOS

        # Make sure the window (with everything now packed inside it) is
        # never taller/wider than the screen -- must run after layout.
        self._fit_window_to_screen()

    # -----------------------------------------------------------------
    # 3. UI construction
    # -----------------------------------------------------------------
    def create_widgets(self, game):
        # --- Scrollable outer container --------------------------------
        # Everything the app packs is placed inside self.scroll_frame rather
        # than directly on the root window. On short/low-resolution screens
        # the stacked toolbars + canvas + composed-level strip can exceed
        # the available screen height; wrapping them in a scrollable canvas
        # means the user can always scroll down to reach controls that
        # don't fit, instead of them being pushed off-screen.
        outer = tk.Frame(self)
        outer.pack(fill=tk.BOTH, expand=True)
        outer.grid_rowconfigure(0, weight=1)
        outer.grid_columnconfigure(0, weight=1)

        self.scroll_canvas = tk.Canvas(outer, highlightthickness=0)
        v_scroll = ttk.Scrollbar(outer, orient=tk.VERTICAL, command=self.scroll_canvas.yview)
        self.scroll_canvas.configure(yscrollcommand=v_scroll.set)
        self.scroll_canvas.grid(row=0, column=0, sticky="nsew")
        v_scroll.grid(row=0, column=1, sticky="ns")

        self.scroll_frame = tk.Frame(self.scroll_canvas)
        self._scroll_frame_window = self.scroll_canvas.create_window(
            (0, 0), window=self.scroll_frame, anchor="nw"
        )
        self.scroll_frame.bind(
            "<Configure>",
            lambda e: self.scroll_canvas.configure(scrollregion=self.scroll_canvas.bbox("all")),
        )
        # Keep the inner frame exactly as wide as the visible canvas so
        # content doesn't get clipped horizontally either.
        self.scroll_canvas.bind(
            "<Configure>",
            lambda e: self.scroll_canvas.itemconfig(self._scroll_frame_window, width=e.width),
        )
        # Mouse-wheel scrolling (Windows/macOS use <MouseWheel>, X11 uses Button-4/5).
        self.scroll_canvas.bind_all("<MouseWheel>", self._on_mousewheel)
        self.scroll_canvas.bind_all("<Button-4>", self._on_mousewheel)
        self.scroll_canvas.bind_all("<Button-5>", self._on_mousewheel)

        root = self.scroll_frame  # everything below packs into the scroll area

        # --- Load buttons ------------------------------------------------
        frame = tk.Frame(root)
        frame.pack(pady=2)
        tk.Button(frame, text="Select Dataset", command=self.load_dataset).pack(side=tk.LEFT, padx=2)
        tk.Button(frame, text="Select Tileset", command=self.load_tileset).pack(side=tk.LEFT, padx=2)
        # A "Load Model" button sat here; removed with the in-browser generation
        # feature (see the commented-out load_model/generate_from_scene methods below).

        # --- Caption/view option checkboxes & toggles ---------------------
        checkbox_frame = tk.Frame(root)
        checkbox_frame.pack(pady=2)

        caption_options_frame = tk.Frame(checkbox_frame)
        caption_options_frame.pack(side=tk.LEFT, padx=5)
        tk.Checkbutton(caption_options_frame, text="Show numeric IDs",
                        variable=self.show_ids, command=self.redraw).pack(anchor=tk.W)
        tk.Checkbutton(caption_options_frame, text="Describe Absence",
                        variable=self.describe_absence).pack(anchor=tk.W)

        tk.Button(checkbox_frame, text="Regenerate Caption", command=self.regenerate_caption).pack(side=tk.LEFT, padx=5)
        tk.Button(checkbox_frame, text="Toggle View Mode", command=self.toggle_view_mode).pack(side=tk.LEFT, padx=5)
        tk.Button(checkbox_frame, text="Toggle A* Path", command=self.toggle_astar_path).pack(side=tk.LEFT, padx=5)

        # Toggle for the 'filter_reason' field carried by entries in the
        # *-filtered datasets (created by create_megaman_json_data.py's
        # apply_filters). Packed dynamically in _update_filter_reason_display:
        # it only appears for entries that actually have the field.
        self.toggle_filter_reason_button = tk.Button(
            checkbox_frame, text="Show Filter Reason", command=self.toggle_filter_reason
        )

        self.play_mmlv_button = tk.Button(
            checkbox_frame, text="Play in Mega Man Maker", command=self.open_mmlv_in_browser
        )

        # Shown only while the toggle above is on; kept in its own always-packed
        # label so toggling its text doesn't shift the rest of the layout.
        self.filter_reason_label = tk.Label(root, text="", fg="red")
        self.filter_reason_label.pack(pady=(0, 2))

        self.show_real_var = tk.BooleanVar(value=False)
        self.show_real_image_button = tk.Checkbutton(
            checkbox_frame, text="Show Real Image", variable=self.show_real_var,
            command=self.show_real_image, state=tk.DISABLED
        )
        self.show_real_image_button.pack(side=tk.LEFT, padx=5)

        # --- Main scene canvas --------------------------------------------
        self.canvas = tk.Canvas(root, bg="white", width=self.window_size, height=self.window_size - 100)
        self.canvas.pack(pady=1)

        # --- Attribute / caption panel -------------------------------------
        caption_area = tk.Frame(root)
        caption_area.pack(pady=2)

        caption_nav_frame = tk.Frame(caption_area)
        caption_nav_frame.pack()
        tk.Label(caption_nav_frame, text="Attribute:").pack(side=tk.LEFT, padx=(5, 2))
        self.attr_var = tk.StringVar()
        self.attr_dropdown = ttk.Combobox(caption_nav_frame, textvariable=self.attr_var, state="readonly", width=24)
        self.attr_dropdown.pack(side=tk.LEFT, padx=(2, 10))
        self.attr_dropdown.bind("<<ComboboxSelected>>", self.on_attr_select)
        self.prev_caption_button = tk.Button(caption_nav_frame, text="<< Prev", command=self.prev_caption)
        self.prev_caption_button.pack(side=tk.LEFT, padx=2)
        self.caption_index_label = tk.Label(caption_nav_frame, text="1 / 1")
        self.caption_index_label.pack(side=tk.LEFT, padx=5)
        self.next_caption_button = tk.Button(caption_nav_frame, text="Next >>", command=self.next_caption)
        self.next_caption_button.pack(side=tk.LEFT, padx=2)

        # insertontime=0 hides the blinking caret so clicking looks like
        # selecting text rather than entering an edit field; takefocus=0
        # keeps it out of Tab traversal.
        self.caption_text = tk.Text(
            caption_area, height=3, width=int(self.window_size / 8), wrap=tk.WORD,
            insertontime=0, takefocus=0
        )
        self.caption_text.pack(pady=2)
        self.caption_text.tag_configure("center", justify="center")
        # Read-only but still selectable/copyable: block edits, not selection/copy.
        self.caption_text.bind("<Key>", lambda e: "break")
        # Arrow keys must still move between samples even when the caption has
        # focus; these more-specific bindings take precedence over <Key> above.
        self.caption_text.bind("<Left>", lambda e: (self.prev_sample(), "break")[1])
        self.caption_text.bind("<Right>", lambda e: (self.next_sample(), "break")[1])
        self.caption_text.bind("<Button-2>", lambda e: "break")  # middle-click paste
        self.caption_text.bind("<Control-v>", lambda e: "break")
        self.caption_text.bind("<Control-V>", lambda e: "break")
        self.caption_text.bind("<Delete>", lambda e: "break")
        self.caption_text.bind("<BackSpace>", lambda e: "break")
        self.caption_text.bind("<Control-c>", self.copy_caption_text)
        self.caption_text.bind("<Control-C>", self.copy_caption_text)
        self.caption_text.bind("<Command-c>", self.copy_caption_text)
        self.caption_text.bind("<Command-C>", self.copy_caption_text)
        self.caption_context_menu = tk.Menu(self, tearoff=0)
        self.caption_context_menu.add_command(label="Copy", command=self.copy_caption_text)
        self.caption_text.bind("<Button-3>", self.show_caption_context_menu)
        self.caption_text.bind("<Control-Button-1>", self.show_caption_context_menu)  # macOS

        # Holds the prompt/caption toggle button below the caption box.
        self.caption_cycle_frame = tk.Frame(root)
        self.caption_cycle_frame.pack(pady=(0, 2))
        self.prompt_toggle_button = tk.Button(
            self.caption_cycle_frame, text="Show Prompt", command=self.toggle_prompt
        )

        # --- Sample navigation ----------------------------------------------
        nav_info_frame = tk.Frame(root)
        nav_info_frame.pack(pady=2)

        self.sample_label = tk.Label(nav_info_frame, text="Sample: 0 / 0")
        self.sample_label.pack(side=tk.LEFT, padx=5)

        tk.Label(nav_info_frame, text="Jump to:").pack(side=tk.LEFT)
        self.jump_entry = tk.Entry(nav_info_frame, width=8)
        self.jump_entry.pack(side=tk.LEFT)
        self.jump_entry.bind("<Return>", self.jump_to_sample)
        # A "Generate From Scene" button and "Steps" field sat here; removed
        # with the model-loading feature (see the commented-out methods below).

        tk.Button(nav_info_frame, text="<< Prev", command=self.prev_sample).pack(side=tk.LEFT, padx=10)
        tk.Button(nav_info_frame, text="Next >>", command=self.next_sample).pack(side=tk.LEFT, padx=10)

        # --- Composed level controls & thumbnails ----------------------------
        self.composed_frame = tk.Frame(root)
        self.composed_frame.pack(pady=(10, 2))

        self.play_composed_button = tk.Button(self.composed_frame, text="Play Composed Level", command=self.play_composed_level)
        self.play_composed_button.pack(side=tk.LEFT, padx=2)
        self.astar_composed_button = tk.Button(self.composed_frame, text="Use A* on Composed Level",
                                                 command=self.astar_composed_level, state=tk.DISABLED)
        self.astar_composed_button.pack(side=tk.LEFT, padx=2)

        self.use_snes_graphics = tk.BooleanVar(value=False)
        self.graphics_checkbox = ttk.Checkbutton(
            self.composed_frame, text="Use SNES Graphics", variable=self.use_snes_graphics, state=tk.DISABLED
        )
        self.graphics_checkbox.pack(side=tk.LEFT, padx=2)

        self.save_composed_button = tk.Button(self.composed_frame, text="Save Composed Level", command=self.save_composed_level)
        self.save_composed_button.pack(side=tk.LEFT, padx=2)

        self.add_to_composed_level_button = tk.Button(self.composed_frame, text="Add To Level", command=self.add_to_composed_level)
        self.add_to_composed_level_button.pack(side=tk.LEFT, padx=2)

        tk.Button(self.composed_frame, text="Move Left", command=self.move_selected_thumbnail_left).pack(side=tk.LEFT, padx=2)
        tk.Button(self.composed_frame, text="Move Right", command=self.move_selected_thumbnail_right).pack(side=tk.LEFT, padx=2)
        tk.Button(self.composed_frame, text="Delete", command=self.delete_selected_thumbnail).pack(side=tk.LEFT, padx=2)
        self.clear_composed_button = tk.Button(self.composed_frame, text="Clear Composed Level", command=self.clear_composed_level)
        self.clear_composed_button.pack(side=tk.LEFT, padx=2)
        self.build_mm_level_button = tk.Button(
            self.composed_frame, text="Build Mega Man Level", command=self.open_megaman_layout_editor, state=tk.DISABLED
        )
        self.build_mm_level_button.pack(side=tk.LEFT, padx=2)
        self.large_view_button = tk.Button(self.composed_frame, text="Large View", command=self.show_large_composed_view)
        self.large_view_button.pack(side=tk.LEFT, padx=2)

        self.composed_thumb_frame = tk.Frame(root)
        self.composed_thumb_frame.pack(fill=tk.X)

        # --- Game selector -----------------------------------------------
        def on_game_select(event=None):
            game_display_var = self.game_display_var.get()
            self.game.set(common_settings.GAME_DISPLAY_MAPPING.get(game_display_var, game_display_var))

            config = common_settings.get_game_config(self.game.get())
            new_tileset_path = config["tileset"]
            tileset_changed = (new_tileset_path and os.path.isfile(new_tileset_path)
                                and new_tileset_path != self.tileset_path)
            if tileset_changed:
                self.tileset_path = new_tileset_path

            self._update_game_specific_controls()

            if tileset_changed and self.dataset_path and self.tileset_path:
                self.load_files_from_paths(self.dataset_path, self.tileset_path)

        game_display = common_settings.GAME_ALIASES[game]
        self.game_display_var = tk.StringVar(value=game_display)
        self.game = tk.StringVar(value=common_settings.GAME_DISPLAY_MAPPING[self.game_display_var.get()])
        self.game_label = ttk.Label(self.composed_frame, text="Select Game:", style="TLabel")
        self.game_label.pack()
        self.game_dropdown = ttk.Combobox(
            self.composed_frame, textvariable=self.game_display_var,
            values=common_settings.GAME_DISPLAY_NAMES, state="readonly"
        )
        self.game_dropdown.pack()
        self.game_dropdown.bind("<<ComboboxSelected>>", on_game_select)
        self._update_game_specific_controls()

    def _on_mousewheel(self, event):
        """Scroll the outer container. Handles Windows/macOS (<MouseWheel>,
        signed event.delta) and X11 (<Button-4>/<Button-5>)."""
        if event.num == 4:
            self.scroll_canvas.yview_scroll(-1, "units")
        elif event.num == 5:
            self.scroll_canvas.yview_scroll(1, "units")
        else:
            self.scroll_canvas.yview_scroll(-1 if event.delta > 0 else 1, "units")

    def _fit_window_to_screen(self):
        """Cap the initial window size to a fraction of the screen so it never
        opens taller or wider than the display -- the scrollable container
        (see create_widgets) then lets the user reach anything that doesn't
        fit. Also makes the window resizable so it can be maximized."""
        self.update_idletasks()
        screen_w = self.winfo_screenwidth()
        screen_h = self.winfo_screenheight()
        max_w = int(screen_w * MAX_SCREEN_FRACTION)
        max_h = int(screen_h * MAX_SCREEN_FRACTION)

        wanted_w = min(self.winfo_reqwidth(), max_w)
        wanted_h = min(self.winfo_reqheight(), max_h)
        # Give the composed-level area and canvas room to breathe even on
        # very short screens, but never exceed the capped height.
        wanted_h = max(wanted_h, min(max_h, 480))

        x = max(0, (screen_w - wanted_w) // 2)
        y = max(0, (screen_h - wanted_h) // 2)
        self.geometry(f"{wanted_w}x{wanted_h}+{x}+{y}")
        self.minsize(480, 320)
        self.resizable(True, True)

    def bind_keys(self):
        self.bind("<Right>", lambda e: self.next_sample())
        self.bind("<Left>", lambda e: self.prev_sample())
        self.bind("<Up>", lambda e: self.prev_caption())
        self.bind("<Down>", lambda e: self.next_caption())

    def _update_game_specific_controls(self):
        """Enable/disable UI controls that only make sense for certain games."""
        is_mm2 = self.game.get() == "MM2"
        is_mario = self.game.get() == "Mario"
        is_megaman = self.game.get() in ("MM-Simple", "MM-Full", "MMLV")

        self.show_real_image_button.config(state=tk.NORMAL if is_mm2 else tk.DISABLED)
        if not is_mm2:
            self.show_real_var.set(False)
            self._exit_real_image_mode()

        self.graphics_checkbox.config(state=tk.NORMAL if is_mario else tk.DISABLED)
        if not is_mario:
            self.use_snes_graphics.set(False)

        # Mario uses the Java sim; MM2 uses the Python A* check in astar_composed_level.
        self.astar_composed_button.config(state=tk.NORMAL if (is_mario or is_mm2) else tk.DISABLED)
        self.build_mm_level_button.config(state=tk.NORMAL if is_megaman else tk.DISABLED)

    # -----------------------------------------------------------------
    # 4. File loading (dataset / tileset)
    # -----------------------------------------------------------------
    def load_files(self):
        dataset_path = filedialog.askopenfilename(title="Select dataset JSON")
        tileset_path = filedialog.askopenfilename(title="Select tileset JSON")
        if not dataset_path or not tileset_path:
            return
        self.load_files_from_paths(dataset_path, tileset_path)

    def load_dataset(self):
        path = filedialog.askopenfilename(title="Select dataset JSON")
        if not path:
            return
        self.dataset_path = path
        if self.dataset_path and self.tileset_path:
            self.load_files_from_paths(self.dataset_path, self.tileset_path)

    def load_tileset(self):
        path = filedialog.askopenfilename(title="Select tileset JSON")
        if not path:
            return
        self.tileset_path = path
        if self.dataset_path and self.tileset_path:
            self.load_files_from_paths(self.dataset_path, self.tileset_path)

    def load_files_from_paths(self, dataset_path, tileset_path):
        print("DATASET =", dataset_path)
        print("TILESET =", tileset_path)

        self.dataset_path = dataset_path
        try:
            with open(dataset_path, 'r', encoding='utf-8') as f:
                self.dataset = json.load(f)

            # Datasets usually hold both scenes and captions, but a bare list of
            # scene grids is converted to dict form with an empty caption.
            if isinstance(self.dataset, list) and all(isinstance(item, list) for item in self.dataset):
                self.dataset = [{'scene': item, 'caption': ''} for item in self.dataset]

            # MarioMakerPCG's multi-caption normalization is left commented out because this viewer
            # pages a scene's captions through the attribute dropdown rather than a 'captions' list.
            #normalized_dataset = []
            #for item in self.dataset:
            #    if isinstance(item, list):
            #        normalized_dataset.append({'scene': item, 'captions': ['']})
            #    else:
            #        captions = [item['caption']] if item.get('caption') else []
            #        idx = 1
            #        while f'caption{idx}' in item:
            #            captions.append(item[f'caption{idx}'])
            #            idx += 1
            #        item['captions'] = captions or ['']
            #        item.setdefault('caption', item['captions'][0])
            #        normalized_dataset.append(item)
            #self.dataset = normalized_dataset

            _, self.id_to_char, self.char_to_id, self.tile_descriptors = extract_tileset(tileset_path)
            self.color_map = self._build_color_map()
            # Start the new dataset at the first sample and its first caption.
            self.current_sample_idx = 0
            self.current_caption_idx = 0
            self.redraw()
        except Exception as e:
            print(f"Error loading files: {e}")
            raise e

    def _build_color_map(self):
        """Build the MM2 tag-based tile color map used by the grid view."""
        color_map = {}
        for tile_id, char in self.id_to_char.items():
            descriptors = self.tile_descriptors.get(char, set())
            color = (0.80, 0.80, 0.80)
            for tag, col in MM2_TAG_COLORS:
                if tag in descriptors:
                    color = col
                    break
            color_map[tile_id] = color
        return color_map

    # -----------------------------------------------------------------
    # 5. Rendering (main canvas)
    # -----------------------------------------------------------------
    def redraw(self):
        if not self.dataset:
            return

        self.canvas.delete("all")
        sample = self.dataset[self.current_sample_idx]

        if isinstance(sample, list):
            # Fallback for datasets that are bare scene grids with no captions.
            sample = {"scene": sample, "caption": "No caption available."}

        # Refresh the dropdown and grab the picked attribute's value(s) to page through.
        attr_names = self._attribute_names(sample)
        self._sync_attr_dropdown(attr_names)
        values = self._selected_values(sample)

        self.current_caption_idx = max(0, min(self.current_caption_idx, len(values) - 1))
        self.caption_index_label.config(text=f"{self.current_caption_idx + 1} / {len(values)}")
        nav_state = tk.NORMAL if len(values) > 1 else tk.DISABLED  # only page when >1 value
        self.prev_caption_button.config(state=nav_state)
        self.next_caption_button.config(state=nav_state)

        # Dynamically size the tile grid/canvas for this scene.
        self.update_tile_and_canvas_size(sample['scene'])

        phrase_colors = self._build_phrase_colors(sample)

        if getattr(self, 'show_real', False) and self._render_real_image():
            pass  # real source image drawn onto the canvas; nothing else to draw
        elif getattr(self, 'show_images', False):
            self._draw_image_view(sample)
        else:
            self._draw_grid_view(sample, phrase_colors)

        # Refresh the filter_reason line for this entry (shown only while the toggle is on).
        self._update_filter_reason_display(sample)
        self._update_mmlv_button(sample)

        self._draw_caption_box(sample, phrase_colors)

        self.sample_label.config(text=f"Sample: {self.current_sample_idx + 1} / {len(self.dataset)}")
        self.title(f"Tile Dataset Viewer - Sample {self.current_sample_idx + 1} / {len(self.dataset)}")

    def _build_phrase_colors(self, sample):
        """Generate unique colors for the caption phrases in `sample['details']`,
        keyed off each game's TOPIC_KEYWORDS list."""
        from captions.caption_match import TOPIC_KEYWORDS  # Mario (default)
        from captions.LR_caption_match import TOPIC_KEYWORDS as LR_TOPIC_KEYWORDS
        from captions.MM_caption_match import TOPIC_KEYWORDS as MM_TOPIC_KEYWORDS
        # MM2 has no fixed topic-keyword list (its captions derive from the
        # tileset), so MM2 falls back to Mario's topic colors below.

        if self.game.get() == "LR":
            TOPIC_KEYWORDS = LR_TOPIC_KEYWORDS
        elif self.game.get() in ("MM-Simple", "MM-Full", "MMLV"):
            TOPIC_KEYWORDS = MM_TOPIC_KEYWORDS
        # else: keep Mario's TOPIC_KEYWORDS (also used as the MM2 fallback)

        topic_colors = {}
        golden_ratio_conjugate = 0.618033988749895
        h = random.random()  # start at a random point on the hue wheel
        for topic in TOPIC_KEYWORDS:
            h = (h + golden_ratio_conjugate) % 1
            saturation = 0.7 + 0.2 * random.random()  # 0.7-0.9
            lightness = 0.45 + 0.1 * random.random()  # 0.45-0.55
            r, g, b = colorsys.hls_to_rgb(h, lightness, saturation)
            topic_colors[topic] = f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}"

        phrase_colors = {}
        if 'details' in sample:
            for phrase in sample['details']:
                for topic in TOPIC_KEYWORDS:
                    if topic in phrase:
                        phrase_colors[phrase] = topic_colors[topic]
                        break  # stop at the first matching topic
        return phrase_colors

    def _draw_image_view(self, sample):
        """Render the sample as a generated tile image (optionally with the A* overlay)."""
        from level_dataset import visualize_samples

        # With the A* overlay on, render the path-annotated image instead of the
        # plain one; fall back to the plain render if the path can't be produced.
        image = None
        if getattr(self, 'show_astar_path', False):
            image = self._astar_overlay_image(sample['scene'])

        if image is None:
            num_classes = {
                "Mario": common_settings.MARIO_TILE_COUNT,
                "LR": common_settings.LR_TILE_COUNT,
                "MM-Simple": common_settings.MM_SIMPLE_TILE_COUNT,
                "MMLV": common_settings.MMLV_TILE_COUNT,
                "MM-Full": common_settings.MM_FULL_TILE_COUNT,
                "MM2": common_settings.MM2_TILE_COUNT,
            }.get(self.game.get(), len(self.id_to_char))  # fallback: size from the tileset

            one_hot_scene = torch.nn.functional.one_hot(
                torch.tensor(sample['scene'], dtype=torch.long), num_classes=num_classes
            ).float().permute(2, 0, 1).unsqueeze(0)
            image = visualize_samples(one_hot_scene, game=self.game.get())

        if isinstance(image, list):
            image = image[0]
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        self.current_pil_image = image  # store for "Save Image As"
        self._blit_image_to_canvas(image)

    def _draw_grid_view(self, sample, phrase_colors):
        """Render the sample as a numeric-ID / character grid, with tile
        backgrounds colored by tag and split into triangles where a cell is
        covered by more than one caption phrase."""
        self.current_pil_image = None  # nothing to save in grid mode
        font = ("Courier", self.font_size)
        color_map = getattr(self, 'color_map', None) or {}
        base_colors = level_dataset.colors()

        height = len(sample['scene'])
        width = len(sample['scene'][0])
        for y in range(height):
            for x in range(width):
                tile_id = sample['scene'][y][x]
                text = str(tile_id) if self.show_ids.get() else self.id_to_char.get(tile_id, '?')

                if self.game.get() == "MM2":
                    # MM2: prefer the tag-based color, then the palette, then neutral gray.
                    if tile_id in color_map:
                        r, g, b = color_map[tile_id]
                    elif tile_id < len(base_colors):
                        r, g, b = base_colors[tile_id]
                    else:
                        r, g, b = (0.80, 0.80, 0.80)
                else:
                    r, g, b = base_colors[tile_id % len(base_colors)]
                color_hex = f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}"

                matching_phrases = []
                if 'details' in sample:
                    for phrase, coords in sample['details'].items():
                        if (y, x) in coords:
                            matching_phrases.append(phrase)

                if not matching_phrases:
                    self.canvas.create_rectangle(
                        x * self.tile_size, y * self.tile_size,
                        (x + 1) * self.tile_size, (y + 1) * self.tile_size,
                        fill="white", outline=""
                    )
                else:
                    triangles = self.create_triangle_coords(x, y, len(matching_phrases))
                    for i, phrase in enumerate(matching_phrases[:4]):  # cap at 4 colors
                        coords = []
                        for point in triangles[i]:
                            coords.extend(point)
                        self.canvas.create_polygon(*coords, fill=phrase_colors[phrase], outline="")

                self.canvas.create_text(
                    x * self.tile_size + self.tile_size // 2,
                    y * self.tile_size + self.tile_size // 2,
                    text=text, font=font, anchor="center", fill=color_hex
                )

    def _draw_caption_box(self, sample, phrase_colors):
        """Fill the text box below the canvas with either the 'prompt' field
        or the currently-picked attribute's value, coloring caption phrases
        by topic when possible."""
        has_prompt = isinstance(sample, dict) and 'prompt' in sample
        self._update_prompt_toggle_control(has_prompt)

        self.caption_text.configure(state="normal")
        self.caption_text.delete("1.0", tk.END)

        if self.show_prompt and has_prompt:
            # prompt has no phrase coords, so just show it plain
            self.caption_text.tag_configure("black", foreground="black")
            self.caption_text.insert(tk.END, str(sample.get('prompt', '')), ("black", "center"))
        else:
            values = self._selected_values(sample)
            caption_text = str(values[self.current_caption_idx])
            for part in caption_text.split('.'):
                part = part.strip()
                if part:
                    part = part + "."  # add the period back
                    color = phrase_colors.get(part, "black")
                    part = part + " "  # add a trailing space for readability
                    self.caption_text.tag_configure(color, foreground=color)
                    self.caption_text.insert(tk.END, part, (color, "center"))

        # Grow the text box to fit the full content so long captions/prompts aren't clipped.
        self._resize_caption_box()
        # Deliberately not disabled, so the user can still select/copy the text.

    def _blit_image_to_canvas(self, image):
        """Scale `image` to fit the canvas (never upscaling) and draw it centered."""
        canvas_width = int(self.canvas['width'])
        canvas_height = int(self.canvas['height'])
        img_width, img_height = image.size
        scale = min(canvas_width / img_width, canvas_height / img_height, 1.0)
        display_image = image
        if scale < 1.0:
            new_size = (int(img_width * scale), int(img_height * scale))
            display_image = image.resize(new_size, Image.Resampling.NEAREST)
        photo_image = PIL.ImageTk.PhotoImage(display_image)
        self.canvas.create_image(canvas_width // 2, canvas_height // 2, image=photo_image, anchor="center")
        self.photo_image = photo_image  # keep a reference to avoid garbage collection

    def create_triangle_coords(self, x, y, num_colors):
        """Coordinates for splitting tile (x, y) into `num_colors` triangles,
        used when a cell matches more than one caption phrase."""
        x1, y1 = x * self.tile_size, y * self.tile_size
        x2, y2 = (x + 1) * self.tile_size, (y + 1) * self.tile_size
        xm, ym = (x1 + x2) / 2, (y1 + y2) / 2

        if num_colors == 2:
            return [
                [(x1, y1), (x2, y1), (x2, y2)],  # upper right triangle
                [(x1, y1), (x1, y2), (x2, y2)],  # lower left triangle
            ]
        elif num_colors == 3:
            return [
                [(x1, y1), (x2, y1), (x2, y2)],  # upper right triangle
                [(x1, y1), (x1, y2), (xm, ym)],  # left triangle
                [(x1, y2), (x2, y2), (xm, ym)],  # bottom triangle
            ]
        elif num_colors == 4:
            return [
                [(x1, y1), (xm, ym), (x2, y1)],  # top triangle
                [(x2, y1), (xm, ym), (x2, y2)],  # right triangle
                [(x2, y2), (xm, ym), (x1, y2)],  # bottom triangle
                [(x1, y2), (xm, ym), (x1, y1)],  # left triangle
            ]
        else:
            return [[(x1, y1), (x2, y1), (x2, y2), (x1, y2)]]  # full square

    def update_tile_and_canvas_size(self, scene):
        """Update tile_size and canvas size so the level fits perfectly inside the window."""
        height = len(scene)
        width = len(scene[0])
        tile_size_h = int(self.window_size // height)
        tile_size_w = int(self.window_size // width)
        self.tile_size = min(tile_size_h, tile_size_w)
        self.canvas.config(width=self.tile_size * width, height=self.tile_size * height)
        self.font_size = max(self.tile_size // 3, 6)  # smaller font relative to tile size

    # --- Real source-image view (MM2 only) ---------------------------------
    def show_real_image(self):
        """Toggle showing the real source image on the main canvas, in place of
        the ASCII grid. When checked, the image is looked up via the 'image'
        path stored in the dataset (a popup is shown ONLY when it can't be
        loaded). When unchecked, restore whatever view was active before."""
        if self.show_real_var.get():
            if not self.dataset:
                self.show_real_var.set(False)
                return
            self._prev_show_images = getattr(self, 'show_images', False)  # for restoring later
            self.show_real = True
            self.show_images = False
        else:
            self._exit_real_image_mode()
        self.redraw()

    def _exit_real_image_mode(self):
        """Leave real-image mode and restore whatever view (grid or generated
        image) was active before it was turned on, keeping the checkbox in sync."""
        self.show_real = False
        self.show_real_var.set(False)
        self.show_images = getattr(self, '_prev_show_images', False)

    def _resolve_image_path(self, image_path):
        """Resolve the (usually relative) 'image' path from a sample to an
        existing file on disk, or return None if it can't be found.

        The path stored in the JSON (e.g. "ds_images\\source_0_0.png") is
        normally relative to the dataset file's directory, but we also try
        the path as-is and relative to the current working directory."""
        if not image_path:
            return None
        image_path = os.path.normpath(image_path)  # normalize Windows-style backslashes
        candidates = []
        if os.path.isabs(image_path):
            candidates.append(image_path)
        else:
            dataset_path = getattr(self, 'dataset_path', None)
            if dataset_path:
                candidates.append(os.path.join(os.path.dirname(os.path.abspath(dataset_path)), image_path))
            candidates.append(os.path.abspath(image_path))
        for candidate in candidates:
            if os.path.isfile(candidate):
                return candidate
        return None

    def _render_real_image(self):
        """Draw the current sample's real source image onto the main canvas, in
        place of the ASCII grid. Returns True on success. On failure, shows an
        error popup, leaves real-image mode, and returns False so the caller can
        fall back to the normal grid/image rendering."""
        sample = self.dataset[self.current_sample_idx]
        image_path = sample.get('image') if isinstance(sample, dict) else None
        resolved = self._resolve_image_path(image_path)

        if resolved is None:
            self._exit_real_image_mode()
            messagebox.showerror(
                "Image not available",
                f"Could not find the real image for this sample.\n\nPath: {image_path}"
            )
            return False

        try:
            image = Image.open(resolved)
        except Exception as e:
            self._exit_real_image_mode()
            messagebox.showerror("Image not available", f"Failed to open the real image:\n{resolved}\n\n{e}")
            return False

        self.current_pil_image = image  # store for "Save Image As"
        self._blit_image_to_canvas(image)
        return True

    # --- A* path overlay -----------------------------------------------------
    def _astar_overlay_image(self, scene):
        """Render scene with its A* path and explored cells.
        Returns a PIL image, or None if the path can't be produced."""
        astar_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "astar")
        if astar_dir not in sys.path:
            sys.path.insert(0, astar_dir)
        try:
            from astar_traversability_check import evaluate
            from astar_path_visualization import render_info
        except Exception as e:
            print(f"Could not import A* path tools: {e}")
            return None

        game = self.game.get()   # "Mario" / "MM2" / "LR" / "MM-Simple" / "MM-Full"
        trav_game = {"Mario": "Mario", "MM2": "MM2", "LR": "LR",
                     "MM-Simple": "MM", "MM-Full": "MM", "MMLV": "MM"}.get(game)
        if trav_game is None:
            return None
        try:
            ok, stats, info = evaluate(trav_game, scene, self.id_to_char,
                                       self.tile_descriptors, 100000, False, visualize=True)
        except Exception as e:
            print(f"A* path failed for this scene: {e}")
            return None
        if info is None:  # e.g. an LR scene with no gold
            print("No A* path to draw for this scene.")
            return None
        print(f"A* path: {'traversable' if ok else 'NOT traversable'}  ({stats})")
        # game doubles as the render-target name render_info expects.
        return render_info(scene, game, info)

    def _astar_path_for_scene(self, scene, spawn=None, orb=None):
        """Run A* on a single scene and return (pil_image_or_None, solved, stats).
        Shared by MegaManLayoutEditor's 'Show A* Path' button. spawn/orb are MM-only
        optional (x, y) cells (the user's placed spawn/exit markers)."""
        astar_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "astar")
        if astar_dir not in sys.path:
            sys.path.insert(0, astar_dir)
        from astar_traversability_check import astar_path_image

        game_name = self.game.get()
        if game_name not in ("Mario", "LR", "MM-Simple", "MM-Full", "MMLV"):
            return None, False, {}
        return astar_path_image(scene, game_name, self.id_to_char, self.tile_descriptors, spawn=spawn, orb=orb)

    def _render_scene_image(self, scene):
        """Render a tile-ID scene to a PIL image for the currently selected game."""
        from level_dataset import visualize_samples
        game = self.game.get()
        config = common_settings.get_game_config(game)
        num_classes = config["tile_count"]
        one_hot = torch.nn.functional.one_hot(
            torch.tensor(scene, dtype=torch.long), num_classes=num_classes
        ).float().permute(2, 0, 1).unsqueeze(0)
        pil_img = visualize_samples(one_hot, game=game)
        return pil_img[0] if isinstance(pil_img, list) else pil_img

    def _resize_caption_box(self):
        """Grow/shrink the caption box so the whole caption is visible without
        scrolling. Uses the widget's wrapped display-line count; no-ops until
        the widget is laid out (e.g. before the window is first mapped), where
        geometry isn't known yet."""
        self.caption_text.update_idletasks()
        if not self.caption_text.winfo_ismapped():
            # Not laid out yet (e.g. the CLI-loaded first sample, before the
            # window is mapped). Retry once the event loop is idle.
            self.caption_text.after(50, self._resize_caption_box)
            return
        try:
            lines = self.caption_text.count("1.0", "end-1c", "displaylines")
        except tk.TclError:
            return
        if not lines:
            return
        n = lines[0] if isinstance(lines, (tuple, list)) else lines
        self.caption_text.configure(height=max(3, int(n)))

    # -----------------------------------------------------------------
    # 6. Attribute / caption panel
    # -----------------------------------------------------------------
    def _attribute_names(self, sample):
        """Keys we can show in the dropdown: skip the reserved ones, keep strings/numbers/lists."""
        if not isinstance(sample, dict):
            return []
        return [k for k, v in sample.items()
                if k not in self.RESERVED_ATTRS and isinstance(v, (str, int, float, bool, list))]

    def _sync_attr_dropdown(self, attr_names):
        """Update the dropdown options, keeping the current pick if it's still
        there, else fall back to a caption field or the first attribute."""
        if list(self.attr_dropdown['values']) != attr_names:
            self.attr_dropdown['values'] = attr_names
        if self.attr_var.get() not in attr_names:
            default = next((k for k in ("caption", "captions") if k in attr_names), None)
            self.attr_var.set(default or (attr_names[0] if attr_names else ""))
            self.current_caption_idx = 0

    def _selected_values(self, sample):
        """The picked attribute as a list of values. Empty list gives one blank so paging still works."""
        value = sample.get(self.attr_var.get()) if isinstance(sample, dict) else None
        if isinstance(value, list):
            return value if value else ['']
        return [value if value is not None else '']

    def on_attr_select(self, event=None):
        self.current_caption_idx = 0  # new attribute, start at the first value
        self.redraw()

    def toggle_prompt(self):
        """Toggle the text box below the scene between its caption and its 'prompt' field."""
        self.show_prompt = not self.show_prompt
        self.redraw()

    def _update_prompt_toggle_control(self, has_prompt):
        """Show the prompt toggle button only when the current scene has a
        'prompt' field; the label reflects what the button switches the box to."""
        if has_prompt:
            if not self.prompt_toggle_button.winfo_ismapped():
                self.prompt_toggle_button.pack(side=tk.LEFT, padx=5)
            self.prompt_toggle_button.config(text="Show Caption" if self.show_prompt else "Show Prompt")
        else:
            self.prompt_toggle_button.pack_forget()

    def regenerate_caption(self):
        print("Regenerating caption...")
        if not self.dataset:
            return
        sample = self.dataset[self.current_sample_idx]

        if self.game.get() == "LR":
            caption, details = lr_assign_caption(
                sample['scene'], self.id_to_char, self.char_to_id, self.tile_descriptors,
                describe_locations=False, describe_absence=self.describe_absence.get(),
                debug=True, return_details=True
            )
        elif self.game.get() in ("MM-Full", "MM-Simple", "MMLV"):
            # Use the entrance/exit direction data baked into the sample itself
            # (populated by create_megaman_json_data.py when run with
            # --direction_captions), instead of parsing it back out of the
            # previous caption text.
            data = sample.get('data', None)
            caption, details = mm_assign_caption(
                sample['scene'], self.id_to_char, self.char_to_id, self.tile_descriptors,
                describe_locations=False, describe_absence=self.describe_absence.get(),
                data=data, debug=True, return_details=True
            )
        elif self.game.get() == "Mario":
            caption, details = assign_caption(
                sample['scene'], self.id_to_char, self.char_to_id, self.tile_descriptors,
                describe_locations=False, describe_absence=self.describe_absence.get(),
                debug=True, return_details=True
            )
        elif self.game.get() == "MM2":
            # Code from interactive_tile_level_generator.
            _, _, ground_chars = get_tile_categories(self.tileset_path)
            char_names = get_char_names(self.tileset_path)
            caption = mm2_assign_caption(sample['scene'], self.id_to_char, char_names, ground_chars)
            details = None  # MM2 assign_caption only returns a single caption string

        sample['caption'] = caption
        sample['captions'] = [caption]
        sample['details'] = details
        # jump the dropdown to the new caption
        self.attr_var.set('caption')
        self.current_caption_idx = 0
        print(f"New caption: {caption}")
        print(details)
        self.redraw()

    def show_caption_context_menu(self, event):
        try:
            self.caption_context_menu.tk_popup(event.x_root, event.y_root)
        finally:
            self.caption_context_menu.grab_release()

    def copy_caption_text(self, event=None):
        try:
            selection = self.caption_text.get(tk.SEL_FIRST, tk.SEL_LAST)
        except tk.TclError:
            selection = self.caption_text.get("1.0", tk.END)  # no selection, copy all
        self.clipboard_clear()
        self.clipboard_append(selection)
        return "break"

    # -----------------------------------------------------------------
    # 7. Sample navigation
    # -----------------------------------------------------------------
    def prev_sample(self):
        if self.current_sample_idx > 0:
            self.current_sample_idx -= 1
            self.current_caption_idx = 0
            self.redraw()

    def next_sample(self):
        if self.current_sample_idx < len(self.dataset) - 1:
            self.current_sample_idx += 1
            self.current_caption_idx = 0
            self.redraw()

    def jump_to_sample(self, event=None):
        try:
            idx = int(self.jump_entry.get()) - 1
            if 0 <= idx < len(self.dataset):
                self.current_sample_idx = idx
                self.current_caption_idx = 0
                self.redraw()
            else:
                print("Index out of range.")
        except ValueError:
            print("Invalid index entered.")

    def prev_caption(self):
        if self.current_caption_idx > 0:
            self.current_caption_idx -= 1
            self.redraw()

    def next_caption(self):
        if not self.dataset:
            return
        values = self._selected_values(self.dataset[self.current_sample_idx])
        if self.current_caption_idx < len(values) - 1:
            self.current_caption_idx += 1
            self.redraw()

    # -----------------------------------------------------------------
    # 8. View-mode toggles
    # -----------------------------------------------------------------
    def toggle_view_mode(self):
        """Toggle between numeric/character grid and image view modes."""
        self.show_real = False  # leaving real-image mode
        self.show_real_var.set(False)
        self.show_images = not getattr(self, 'show_images', False)
        self.redraw()

    def toggle_astar_path(self):
        """Toggle the A* path overlay on the current level. The overlay only
        makes sense on the image view, so turning it on forces image mode."""
        self.show_astar_path = not getattr(self, 'show_astar_path', False)
        if self.show_astar_path:
            self.show_images = True
        self.redraw()

    def toggle_filter_reason(self):
        """Toggle display of the current entry's 'filter_reason' field (present
        on entries from the *-filtered datasets)."""
        self.show_filter_reason = not getattr(self, 'show_filter_reason', False)
        self.toggle_filter_reason_button.config(
            text="Hide Filter Reason" if self.show_filter_reason else "Show Filter Reason"
        )
        self.redraw()

    def _update_filter_reason_display(self, sample):
        """Show the filter-reason toggle button and line only for entries that carry filter-reason info."""
        reasons = None
        if isinstance(sample, dict):
            reasons = sample.get('filter_reasons')
            if reasons is None:
                single = sample.get('filter_reason')  # backward-compat: old single-reason entries
                reasons = [single] if single is not None else None
        if not reasons:
            self.toggle_filter_reason_button.pack_forget()
            self.filter_reason_label.config(text="")
            return
        if not self.toggle_filter_reason_button.winfo_ismapped():
            self.toggle_filter_reason_button.pack(side=tk.LEFT, padx=5)
        if getattr(self, 'show_filter_reason', False):
            label = "Filter reasons" if len(reasons) != 1 else "Filter reason"
            self.filter_reason_label.config(text=f"{label}: {', '.join(reasons)}")
        else:
            self.filter_reason_label.config(text="")

    def _update_mmlv_button(self, sample):
        """Show the 'Play in Mega Man Maker' button only when the current game
        is a Mega Man variant and the current sample carries an 'mmlvID' field."""
        is_mm_game = self.game.get() in ("MM-Simple", "MM-Full", "MMLV")
        mmlv_id = sample.get('mmlvID') if isinstance(sample, dict) else None
        if is_mm_game and mmlv_id is not None:
            if not self.play_mmlv_button.winfo_ismapped():
                self.play_mmlv_button.pack(side=tk.LEFT, padx=5)
        else:
            self.play_mmlv_button.pack_forget()

    def open_mmlv_in_browser(self):
        sample = self.dataset[self.current_sample_idx]
        mmlv_id = sample.get('mmlvID') if isinstance(sample, dict) else None
        if mmlv_id is None:
            return
        try:
            os.startfile(f"megamaker://?level={mmlv_id}")
        except Exception as e:
            # Fall back to the browser version if the app/protocol isn't available.
            print(f"Could not launch Mega Man Maker directly ({e}); opening in browser instead.")
            webbrowser.open(f"https://megamanmaker.com/?level={mmlv_id}")

    # -----------------------------------------------------------------
    # 9. Composed level management
    # -----------------------------------------------------------------
    @property
    def composed_scenes(self):
        """Read-only view of the scenes currently added to the composed level, in
        the same order as composed_thumbnails. Lets MegaManLayoutEditor treat this
        browser the same way it treats CaptionBuilder's composed_scenes list."""
        return [self.dataset[i]['scene'] for i in self.added_sample_indexes]

    def add_to_composed_level(self):
        idx = self.current_sample_idx
        self.added_sample_indexes.append(idx)
        photo = self._make_thumbnail(self.dataset[idx]['scene'])
        self.composed_thumbnails.append(photo)  # prevent GC
        self.redraw_composed_thumbnails()

    def _make_thumbnail(self, scene):
        """Render a scene to a small PhotoImage suitable for the composed-level strip."""
        from level_dataset import visualize_samples
        one_hot_scene = torch.nn.functional.one_hot(
            torch.tensor(scene, dtype=torch.long), num_classes=len(self.id_to_char)
        ).float().permute(2, 0, 1).unsqueeze(0)
        image = visualize_samples(one_hot_scene, game=self.game.get())
        if isinstance(image, list):
            image = image[0]
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        thumb = image.copy()
        thumb.thumbnail((64, 64), Image.Resampling.NEAREST)
        return PIL.ImageTk.PhotoImage(thumb)

    def redraw_composed_thumbnails(self):
        for widget in self.composed_thumb_frame.winfo_children():
            widget.destroy()
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
            if self.selected_thumb_idx >= len(self.composed_thumbnails):
                self.selected_thumb_idx = len(self.composed_thumbnails) - 1
            if self.selected_thumb_idx < 0:
                self.selected_thumb_idx = None
            self.redraw_composed_thumbnails()

    def move_selected_thumbnail_left(self):
        idx = self.selected_thumb_idx
        if idx is not None and idx > 0:
            self._swap_thumbnails(idx, idx - 1)
            self.selected_thumb_idx -= 1
            self.redraw_composed_thumbnails()

    def move_selected_thumbnail_right(self):
        idx = self.selected_thumb_idx
        if idx is not None and idx < len(self.added_sample_indexes) - 1:
            self._swap_thumbnails(idx, idx + 1)
            self.selected_thumb_idx += 1
            self.redraw_composed_thumbnails()

    def _swap_thumbnails(self, i, j):
        self.added_sample_indexes[i], self.added_sample_indexes[j] = (
            self.added_sample_indexes[j], self.added_sample_indexes[i]
        )
        self.composed_thumbnails[i], self.composed_thumbnails[j] = (
            self.composed_thumbnails[j], self.composed_thumbnails[i]
        )

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

    def edit_composed_scene(self, idx, extra_on_save=None):
        """Open the LevelEditor for a scene in the composed level strip.

        Writes the edit back into self.dataset (this browser stores composed
        scenes as indexes into the dataset rather than a separate list) and
        refreshes that thumbnail. extra_on_save, if given, is called with the
        updated scene afterward - used by MegaManLayoutEditor to refresh its
        own grid render."""
        dataset_idx = self.added_sample_indexes[idx]
        scene = self.dataset[dataset_idx]['scene']
        editor_window = tk.Toplevel(self)
        editor_window.title("Level Editor")

        def on_save(updated_scene):
            self.dataset[dataset_idx]['scene'] = updated_scene
            self.composed_thumbnails[idx] = self._make_thumbnail(updated_scene)
            self.redraw_composed_thumbnails()
            if extra_on_save:
                extra_on_save(updated_scene)

        LevelEditor(
            editor_window, scene, self.id_to_char, self.char_to_id,
            self.tile_descriptors, self.game.get(), on_save=on_save
        )

    def open_megaman_layout_editor(self):
        if self.game.get() not in ("MM-Simple", "MM-Full", "MMLV"):
            messagebox.showinfo("Mega Man only", "Switch the game dropdown to a Mega Man mode to use this tool.")
            return
        if not self.added_sample_indexes:
            messagebox.showinfo(
                "No scenes yet",
                "Use 'Add To Level' on one or more scenes first, then open this tool to arrange them."
            )
            return
        MegaManLayoutEditor(self, self)

    def show_large_composed_view(self):
        """Pop up a large rendering of the full composed level, optionally with
        the A* path overlaid if 'Toggle A* Path' is currently on."""
        scene = self.merge_selected_scenes()
        if not scene:
            messagebox.showinfo("No composed level", "Add at least one image to the composed level first.")
            return

        pil_img = None
        if getattr(self, 'show_astar_path', False):
            pil_img = self._astar_overlay_image(scene)  # None if A* fails/can't produce a path
        if pil_img is None:
            pil_img = self._render_scene_image(scene)

        self._show_image_popup(pil_img, "Composed Level - Large View")

    def _show_image_popup(self, pil_img, title):
        """Show a (possibly large) PIL image in a scrollable popup window, capped
        to a sensible fraction of the screen so it doesn't overflow either."""
        win = tk.Toplevel(self)
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

        photo = PIL.ImageTk.PhotoImage(pil_img)
        canvas._photo_ref = photo  # keep a reference so it isn't garbage-collected
        canvas.create_image(0, 0, image=photo, anchor="nw")
        canvas.configure(scrollregion=(0, 0, pil_img.width, pil_img.height))

        screen_w = win.winfo_screenwidth()
        screen_h = win.winfo_screenheight()
        max_w = int(screen_w * MAX_SCREEN_FRACTION)
        max_h = int(screen_h * MAX_SCREEN_FRACTION)
        win.geometry(f"{min(pil_img.width + 24, max_w)}x{min(pil_img.height + 24, max_h)}")

    # -----------------------------------------------------------------
    # 10. Playback & export
    # -----------------------------------------------------------------
    def get_sample_output(self, scene, use_snes_graphics=False):
        if self.game.get() == 'LR':
            level = SampleOutput(level=scene, use_snes_graphics=use_snes_graphics)
        elif self.game.get() == 'Mario':
            if use_snes_graphics is None:
                use_snes_graphics = self.use_snes_graphics.get()
            char_grid = scene_to_ascii(scene, self.id_to_char)
            level = SampleOutput(level=char_grid, use_snes_graphics=use_snes_graphics)
        else:
            # No Java simulator for MM2 or the Mega Man variants (they use the Python A* path).
            raise ValueError(f"get_sample_output: no simulator for game {self.game.get()!r}")
        return level

    def play_composed_level(self):
        scene = self.merge_selected_scenes()
        if not scene:
            return
        # Mario Maker exports a .swe into SMM:WE's level folder and launches the game.
        if self.game.get() == "MM2":
            self._play_composed_swe()
            return
        if self.game.get() == "LR" and not self.validate_lode_runner_level(scene):
            print("Invalid Lode Runner level. Cannot play.")
            return
        level = self.get_sample_output(scene, use_snes_graphics=self.use_snes_graphics.get())
        level.play(
            game=self.game.get(),
            level_idx=(self.added_sample_indexes[0] + 1) if self.added_sample_indexes else 1,
            dataset_path=self.dataset_path if hasattr(self, 'dataset_path') else None
        )

    def astar_composed_level(self):
        scene = self.merge_selected_scenes()
        if not scene:
            return
        if self.game.get() == "MM2":
            # No Java sim for MM2; use the Python astar/ check instead.
            from astar.astar_traversability_check import astar_console_report
            print(astar_console_report(scene, id_to_char=self.id_to_char, tile_descriptors=self.tile_descriptors))
            return
        level = self.get_sample_output(scene, use_snes_graphics=self.use_snes_graphics.get())
        print(level.run_astar())

    def validate_lode_runner_level(self, scene):
        width = len(scene[0])
        for row in scene:
            if len(row) != width:
                print("Level is not rectangular!")
                return False

        if len(scene) != 32 or width != 32:
            print(f"Level is not 32x32! Got {len(scene)}x{width}")
            return False

        player_found = any(self.id_to_char[tile] == 'M' for row in scene for tile in row)
        if not player_found:
            print("No player spawn found!")
            return False

        gold_found = any(self.id_to_char[tile] == 'G' for row in scene for tile in row)
        if not gold_found:
            print("No gold found!")
            return False

        # TODO: could also check for at least one valid move for the player.
        print("Level validation passed.")
        return True

    def save_composed_level(self):
        scene = self.merge_selected_scenes()
        if not scene:
            print("No composed scene to save.")
            return
        # Mario Maker saves a .swe into SMM:WE's level folder instead of a .txt.
        if self.game.get() == "MM2":
            self._save_composed_swe()
            return

        initial_dir = os.path.join(os.getcwd(), "Composed Levels")  # always relative to cwd
        os.makedirs(initial_dir, exist_ok=True)
        file_path = filedialog.asksaveasfilename(
            defaultextension=".txt", filetypes=[("Text files", "*.txt")],
            title="Save Composed Level As", initialdir=initial_dir
        )
        if not file_path:
            print("Save operation cancelled.")
            return
        char_grid = scene_to_ascii(scene, self.id_to_char)
        try:
            with open(file_path, "w") as f:
                for line in char_grid:
                    f.write(line + "\n")
            print(f"Composed level saved to {file_path}")
        except Exception as e:
            print(f"Failed to save composed level: {e}")

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
            filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg"), ("All files", "*.*")],
            title="Save Image As", initialfile=default_filename
        )
        if not file_path:
            return
        try:
            self.current_pil_image.save(file_path)
            messagebox.showinfo("Success", f"Image saved successfully to:\n{file_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save image:\n{str(e)}")

    # --- SMM:WE (.swe) export, MM2 only --------------------------------------
    def _smmwe_niveles_dir(self):
        """SMM:WE's level folder, %LOCALAPPDATA%\\SMM_WE\\Niveles. Falls back to
        a local folder when LOCALAPPDATA isn't set (non-Windows)."""
        base = os.environ.get("LOCALAPPDATA")
        if base:
            return os.path.join(base, "SMM_WE", "Niveles")
        return os.path.join(os.getcwd(), "Niveles")

    def _smmwe_exe_search_paths(self):
        """Candidate paths where SMM:WE may be installed."""
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
        from datetime import datetime
        from mm2pipeline_data.ascii import ascii_to_level
        from mm2pipeline_data.swe import build_world, detect_smmwe_user, encode_swe

        # Keep the full scene (no 15-row A* trim). '_' is padding, not a real
        # tile, but the converter reads it as Goal Ground, so treat it as empty space.
        char_grid = scene_to_ascii(self.merge_selected_scenes(), self.id_to_char, shorten=False)
        ascii_text = "\n".join(row.replace("_", " ") for row in char_grid)
        level_json = ascii_to_level(ascii_text, source_file=name)

        now = datetime.now()
        s0, dropped = build_world(
            level_json, user=detect_smmwe_user(), name=name, desc=None,
            date_str=now.strftime("%d/%m/%Y"), time_str=now.strftime("%H:%M"),
        )
        return encode_swe({"S0": s0, "SB1": {"S1": []}}), dropped

    @staticmethod
    def _report_dropped(dropped):
        if dropped:
            total = sum(dropped.values())
            summary = ", ".join(f"{n}x {nm}" for nm, n in sorted(dropped.items(), key=lambda kv: -kv[1]))
            print(f"  dropped {total} object(s) with no SMM:WE equivalent: {summary}")

    def _save_composed_swe(self):
        """Save the composed scene as a .swe, prompting for a name in Niveles."""
        niveles_dir = self._smmwe_niveles_dir()
        os.makedirs(niveles_dir, exist_ok=True)
        file_path = filedialog.asksaveasfilename(
            defaultextension=".swe", filetypes=[("SMM:WE level", "*.swe")],
            title="Save Composed Level to SMM:WE", initialdir=niveles_dir,
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

        name = "composed_level"
        niveles_dir = self._smmwe_niveles_dir()
        os.makedirs(niveles_dir, exist_ok=True)
        swe_bytes, dropped = self._compose_swe_bytes(name)
        out_path = os.path.join(niveles_dir, name + ".swe")
        with open(out_path, "wb") as f:
            f.write(swe_bytes)
        print(f"Composed level exported to {out_path} ({len(swe_bytes)} bytes)")
        self._report_dropped(dropped)

        exe = self._smmwe_exe_path()
        if exe is None:
            search_paths = self._smmwe_exe_search_paths()
            search_text = "\n".join(search_paths)
            messagebox.showerror(
                "SMM:WE executable not found",
                "Could not find the SMM:WE executable.\n"
                "SMM_WE.exe was searched for in the following locations:\n\n"
                f"{search_text}\n\n"
                "Please install SMM:WE or place SMM_WE.exe in one of these folders."
            )
            print("SMM:WE executable not found. Looked in the following locations:")
            for path in search_paths:
                print(f"  {path}")
            return
        subprocess.Popen([exe], cwd=os.path.dirname(exe))  # run from the install dir so it finds data.win
        print(f"Launched SMM:WE -- open the level browser and play '{name}'.")

    # -----------------------------------------------------------------
    # 11. App lifecycle
    # -----------------------------------------------------------------
    def on_close(self):
        self.destroy()
        sys.exit(0)


# ---------------------------------------------------------------------------
# 12. Command-line entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    dataset_path = None
    if len(sys.argv) >= 2:
        dataset_path = sys.argv[1]
        if not os.path.isfile(dataset_path):
            print("Invalid dataset path provided.")
            dataset_path = None

    game = sys.argv[2] if len(sys.argv) == 3 else "Mario"
    print(f"Game is {game}")

    app = TileViewer(dataset_path, game)
    app.protocol("WM_DELETE_WINDOW", app.on_close)
    app.mainloop()