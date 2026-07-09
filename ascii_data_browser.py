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
from MegaManLayoutEditor import LevelEditor, MegaManLayoutEditor
#from LodeRunner.loderunner.graphics import *
import webbrowser


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
        # Jacob: I think MariobMakerPCG renamed caption_cycle_idx to current_caption_idx
        self.current_caption_idx = 0
        #self.caption_cycle_idx = 0  # which caption field is shown when a scene has several
        self.show_prompt = False  # text box shows the scene's 'prompt' field instead of caption
        self.show_ids = tk.BooleanVar(value=False)
        self.describe_absence = tk.BooleanVar(value=False)
        self.show_images = False        # image view vs numeric/character grid
        self.show_astar_path = False    # overlay the A* path on the image view
        self.show_filter_reason = False # show each entry's 'filter_reason' field (from *-filtered datasets)

        # UI
        self.create_widgets()
        self.bind_keys()

        self.dataset_path = dataset_path
        self.tileset_path = tileset_path

        # Optional initial load from command-line
        if self.dataset_path and self.tileset_path:
            self.load_files_from_paths(self.dataset_path, self.tileset_path)

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
        elif self.game.get()=="MM-Full" or self.game.get()=="MM-Simple" or self.game.get()=="MMLV":
            # Use the entrance/exit direction data baked into the sample itself
            # (populated by create_megaman_json_data.py when run with --direction_captions),
            # instead of parsing it back out of the previous caption text.
            data = sample.get('data', None)

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
        elif self.game.get()=="Mario":
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
        elif self.game.get()=="MM2" or self.game.get()=="MM":
            caption, details = None, None
            # Jacob: There didn't seem to be a way of assigning deterministic Mario Maker captions

        sample['caption'] = caption
        sample['captions'] = [caption]
        sample['details'] = details
        # Jacob: favor current_caption_idx instead of caption_cycle_idx
        self.current_caption_idx = 0
        # self.caption_cycle_idx = 0 
        print(f"New caption: {caption}")
        print(details)
        self.redraw()

    def toggle_view_mode(self):
        """Toggle between numeric/character grid and image view modes."""
        self.show_real = False  # leaving real-image mode
        self.show_real_var.set(False)
        self.show_images = not getattr(self, 'show_images', False)
        self.redraw()

    def toggle_astar_path(self):
        """Toggle the A* path overlay on the current level. The overlay only makes sense
        on the image view, so turning it on forces image mode"""
        self.show_astar_path = not getattr(self, 'show_astar_path', False)
        if self.show_astar_path:
            self.show_images = True
        self.redraw()

    def toggle_filter_reason(self):
        """Toggle display of the current entry's 'filter_reason' field (present on entries
        from the *-filtered datasets)."""
        self.show_filter_reason = not getattr(self, 'show_filter_reason', False)
        self.toggle_filter_reason_button.config(
            text="Hide Filter Reason" if self.show_filter_reason else "Show Filter Reason"
        )
        self.redraw()

    def _update_mmlv_button(self, sample):
        """Show the 'Play in Mega Man Maker' button only when the current game is a
        Mega Man variant and the current sample carries an 'mmlvID' field."""
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
            # Fall back to the browser version if the app/protocol isn't available
            print(f"Could not launch Mega Man Maker directly ({e}); opening in browser instead.")
            url = f"https://megamanmaker.com/?level={mmlv_id}"
            webbrowser.open(url)

    def _update_filter_reason_display(self, sample):
        """Show the filter-reason toggle button and line only for entries that carry filter-reason info"""
        reasons = None
        if isinstance(sample, dict):
            reasons = sample.get('filter_reasons')
            if reasons is None:
                single = sample.get('filter_reason')  # backward-compat: old single-reason entries
                reasons = [single] if single is not None else None
        if not reasons:
            # No field on this entry: hide the button and clear the line.
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

    def _astar_overlay_image(self, scene):
        """
        Render scene with its A* path and explored cells 
        Returns a PIL image, or None if the path can't be produced
        """
        astar_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "astar")
        if astar_dir not in sys.path:
            sys.path.insert(0, astar_dir)
        try:
            from astar_traversability_check import evaluate
            from astar_path_visualization import render_info
        except Exception as e:
            print(f"Could not import A* path tools: {e}")
            return None

        game = self.game.get()   # "Mario" / "LR" / "MM-Simple" / "MM-Full"
        trav_game = {"Mario": "Mario", "LR": "LR",
                     "MM-Simple": "MM", "MM-Full": "MM", "MMLV": "MM"}.get(game)
        if trav_game is None:
            return None
        try:
            ok, stats, info = evaluate(trav_game, scene, self.id_to_char,
                                       self.tile_descriptors, 100000, False, visualize=True)
        except Exception as e:
            print(f"A* path failed for this scene: {e}")
            return None
        if info is None:                 # e.g. an LR scene with no gold
            print("No A* path to draw for this scene.")
            return None
        print(f"A* path: {'traversable' if ok else 'NOT traversable'}  ({stats})")
        # game doubles as the render-target name render_info expects.
        return render_info(scene, game, info)

    @property
    def composed_scenes(self):
        """Read-only view of the scenes currently added to the composed level, in
        the same order as composed_thumbnails. Lets MegaManLayoutEditor treat this
        browser the same way it treats CaptionBuilder's composed_scenes list."""
        return [self.dataset[i]['scene'] for i in self.added_sample_indexes]

    def _render_scene_image(self, scene):
        """Render a tile-ID scene to a PIL image for the currently selected game."""
        from level_dataset import visualize_samples
        game = self.game.get()
        game_to_num_classes = {
            "LR": common_settings.LR_TILE_COUNT,
            "MM-Simple": common_settings.MM_SIMPLE_TILE_COUNT,
            "MM-Full": common_settings.MM_FULL_TILE_COUNT,
            "MMLV": common_settings.MMLV_TILE_COUNT,
            "MM2": common_settings.MM2_TILE_COUNT,
            "Mario": common_settings.MARIO_TILE_COUNT,
        }
        num_classes = game_to_num_classes.get(game, len(self.id_to_char))
        one_hot = torch.nn.functional.one_hot(
            torch.tensor(scene, dtype=torch.long),
            num_classes=num_classes
        ).float().permute(2, 0, 1).unsqueeze(0)
        pil_img = visualize_samples(one_hot, game=game)
        return pil_img[0] if isinstance(pil_img, list) else pil_img

    def edit_composed_scene(self, idx, extra_on_save=None):
        """Open the LevelEditor for a scene in the composed level strip.

        Writes the edit back into self.dataset (this browser stores composed scenes
        as indexes into the dataset rather than a separate list) and refreshes that
        thumbnail. extra_on_save, if given, is called with the updated scene afterward
        - used by MegaManLayoutEditor to refresh its own grid render."""
        dataset_idx = self.added_sample_indexes[idx]
        scene = self.dataset[dataset_idx]['scene']
        editor_window = tk.Toplevel(self)
        editor_window.title("Level Editor")

        def on_save(updated_scene):
            self.dataset[dataset_idx]['scene'] = updated_scene

            rendered = self._render_scene_image(updated_scene)
            thumb = rendered.copy()
            thumb.thumbnail((64, 64), Image.Resampling.NEAREST)
            photo = PIL.ImageTk.PhotoImage(thumb)
            self.composed_thumbnails[idx] = photo
            self.redraw_composed_thumbnails()

            if extra_on_save:
                extra_on_save(updated_scene)

        LevelEditor(
            editor_window,
            scene,
            self.id_to_char,
            self.char_to_id,
            self.tile_descriptors,
            self.game.get(),
            on_save=on_save
        )

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
        return astar_path_image(scene, game_name, self.id_to_char, self.tile_descriptors,
                                spawn=spawn, orb=orb)

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
    
    # Jacob: Next few methods are from MarioMakerPCG. Were they needed?
    def _resolve_image_path(self, image_path):
        """Resolve the (usually relative) 'image' path from a sample to an
        existing file on disk, or return None if it can't be found.

        The path stored in the JSON (e.g. "ds_images\\source_0_0.png") is
        normally relative to the dataset file's directory, but we also try the
        path as-is and relative to the current working directory.
        """
        if not image_path:
            return None
        # Normalize separators so Windows-style backslashes work everywhere.
        image_path = os.path.normpath(image_path)
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

    def show_real_image(self):
        """Toggle showing the real source image on the main canvas, in place of
        the ASCII grid. When checked, the image is looked up via the 'image'
        path stored in the dataset (a popup is shown ONLY when it can't be
        loaded). When unchecked, restore whatever view was active before.
        """
        if self.show_real_var.get():
            if not self.dataset:
                self.show_real_var.set(False)
                return
            # Remember the view we're leaving so we can return to it.
            self._prev_show_images = getattr(self, 'show_images', False)
            self.show_real = True
            self.show_images = False
        else:
            # Restore the view that was active before showing the real image.
            self._exit_real_image_mode()
        self.redraw()

    def _exit_real_image_mode(self):
        """Leave real-image mode and restore whatever view (grid or generated
        image) was active before it was turned on, keeping the checkbox in sync."""
        self.show_real = False
        self.show_real_var.set(False)
        self.show_images = getattr(self, '_prev_show_images', False)

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
                "Could not find the real image for this sample.\n\n"
                f"Path: {image_path}"
            )
            return False

        try:
            image = Image.open(resolved)
        except Exception as e:
            self._exit_real_image_mode()
            messagebox.showerror(
                "Image not available",
                f"Failed to open the real image:\n{resolved}\n\n{e}"
            )
            return False

        self.current_pil_image = image  # Store for "Save Image As".
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
        self.photo_image = photo_image  # Keep a reference to avoid GC.
        return True

    def create_widgets(self):
        frame = tk.Frame(self)
        frame.pack(pady=2)  # Reduced padding for tighter vertical spacing

        load_dataset_button = tk.Button(frame, text="Select Dataset", command=self.load_dataset)
        load_dataset_button.pack(side=tk.LEFT, padx=2)

        load_tileset_button = tk.Button(frame, text="Select Tileset", command=self.load_tileset)
        load_tileset_button.pack(side=tk.LEFT, padx=2)

        # Add a button to load a trained diffusion model
        # Jacob: This never really worked and clutters the interface, so I'm commenting it out.
        #self.load_model_button = tk.Button(frame, text="Load Model", command=self.load_model)
        #self.load_model_button.pack(pady=2)

        checkbox_frame = tk.Frame(self)
        checkbox_frame.pack(pady=2)  # Reduced padding for tighter vertical spacing
        
        # Create a sub-frame for the caption options
        caption_options_frame = tk.Frame(checkbox_frame)
        caption_options_frame.pack(side=tk.LEFT, padx=5)
        
        # Add checkboxes for caption generation options
        tk.Checkbutton(caption_options_frame, text="Show numeric IDs", variable=self.show_ids, command=self.redraw).pack(anchor=tk.W)
        tk.Checkbutton(caption_options_frame, text="Describe Absence", variable=self.describe_absence).pack(anchor=tk.W)
        
        regenerate_button = tk.Button(checkbox_frame, text="Regenerate Caption", command=self.regenerate_caption)
        regenerate_button.pack(side=tk.LEFT, padx=5)

        toggle_view_button = tk.Button(checkbox_frame, text="Toggle View Mode", command=self.toggle_view_mode)
        toggle_view_button.pack(side=tk.LEFT, padx=5)

        toggle_astar_button = tk.Button(checkbox_frame, text="Toggle A* Path", command=self.toggle_astar_path)
        toggle_astar_button.pack(side=tk.LEFT, padx=5)

        # Toggle for revealing the 'filter_reason' field carried by entries in the
        # *-filtered datasets (created by create_megaman_json_data.py's apply_filters).
        # Packed dynamically in _update_filter_reason_display: it only appears for entries
        # that actually have a 'filter_reason' field.
        self.toggle_filter_reason_button = tk.Button(
            checkbox_frame, text="Show Filter Reason", command=self.toggle_filter_reason
        )

        self.play_mmlv_button = tk.Button(
            checkbox_frame, text="Play in Mega Man Maker", command=self.open_mmlv_in_browser
        )

        # Line showing the current entry's filter_reason, only while the toggle above is on.
        # Kept in its own always-packed label so toggling its text doesn't shift the layout.
        self.filter_reason_label = tk.Label(self, text="", fg="red")
        self.filter_reason_label.pack(pady=(0, 2))

        self.show_real_var = tk.BooleanVar(value=False)
        self.show_real_image_button = tk.Checkbutton(
            checkbox_frame,
            text="Show Real Image",
            variable=self.show_real_var,
            command=self.show_real_image,
            state=tk.DISABLED
        )
        self.show_real_image_button.pack(side=tk.LEFT, padx=5)

        self.canvas = tk.Canvas(self, bg="white", width=self.window_size, height=self.window_size - 100)  # Further reduced height to minimize empty space
        self.canvas.pack(pady=1)  # Reduced padding for tighter vertical spacing

        caption_nav_frame = tk.Frame(self)
        caption_nav_frame.pack(pady=2)
        tk.Button(caption_nav_frame, text="< Caption", command=self.prev_caption).pack(side=tk.LEFT, padx=2)
        self.caption_index_label = tk.Label(caption_nav_frame, text="Caption 1 / 1")
        self.caption_index_label.pack(side=tk.LEFT, padx=5)
        tk.Button(caption_nav_frame, text="Caption >", command=self.next_caption).pack(side=tk.LEFT, padx=2)

        # Add Text widget for captions. insertontime=0 hides the blinking insert caret so
        # clicking the caption looks like selecting text, not entering an edit field;
        # takefocus=0 keeps it out of Tab traversal.
        self.caption_text = tk.Text(
            self, height=3, width=int(self.window_size / 8), wrap=tk.WORD,
            insertontime=0, takefocus=0
        )
        self.caption_text.pack(pady=2)
        self.caption_text.tag_configure("center", justify="center")
        # Read-only but still selectable/copyable, so block edits but not selection/copy.
        self.caption_text.bind("<Key>", lambda e: "break")
        # Arrow keys must still move between samples even when the caption has focus; the
        # more-specific Left/Right bindings take precedence over the <Key> handler above.
        self.caption_text.bind("<Left>", lambda e: (self.prev_sample(), "break")[1])
        self.caption_text.bind("<Right>", lambda e: (self.next_sample(), "break")[1])
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

        # Button to cycle between multiple caption fields on a scene (e.g. caption,
        # caption1, ... from llm_ascii_to_caption). The frame stays packed in place so the
        # layout is stable; the button/label inside are shown only for multi-caption scenes.
        self.caption_cycle_frame = tk.Frame(self)
        self.caption_cycle_frame.pack(pady=(0, 2))
        self.caption_cycle_button = tk.Button(
            self.caption_cycle_frame, text="Cycle Caption", command=self.cycle_caption
        )
        self.caption_cycle_label = tk.Label(self.caption_cycle_frame, text="")

        # Button to switch the text box between the scene's caption and its optional
        # 'prompt' field. Like the cycle controls, it stays hidden unless the current scene
        # actually has a 'prompt' field.
        self.prompt_toggle_button = tk.Button(
            self.caption_cycle_frame, text="Show Prompt", command=self.toggle_prompt
        )

        # Combined navigation and info frame
        nav_info_frame = tk.Frame(self)
        nav_info_frame.pack(pady=2)  # Place above composed controls and thumbnails

        # Sample info and jump
        self.sample_label = tk.Label(nav_info_frame, text="Sample: 0 / 0")
        self.sample_label.pack(side=tk.LEFT, padx=5)

        tk.Label(nav_info_frame, text="Jump to:").pack(side=tk.LEFT)
        self.jump_entry = tk.Entry(nav_info_frame, width=8)
        self.jump_entry.pack(side=tk.LEFT)
        self.jump_entry.bind("<Return>", self.jump_to_sample)

        # Generate button (initially disabled)
        # Jacob: This never really worked and clutters the interface, so I'm commenting it out.
        #self.generate_button = tk.Button(nav_info_frame, text="Generate From Scene", command=self.generate_from_scene, state=tk.DISABLED)
        #self.generate_button.pack(side=tk.LEFT, padx=20)

        # Steps input field
        # Jacob: This is related to the Generate From Scene button, which is disabled, so I'm disabling this too.
        #tk.Label(nav_info_frame, text="Steps:").pack(side=tk.LEFT)
        #self.steps_entry = tk.Entry(nav_info_frame, width=4)
        #self.steps_entry.insert(0, "50")  # Default value
        #self.steps_entry.config(state=tk.DISABLED)  # Initially disabled
        #self.steps_entry.pack(side=tk.LEFT, padx=20)

        # Navigation buttons
        tk.Button(nav_info_frame, text="<< Prev", command=self.prev_sample).pack(side=tk.LEFT, padx=10)
        tk.Button(nav_info_frame, text="Next >>", command=self.next_sample).pack(side=tk.LEFT, padx=10)

        # Composed level controls (below navigation)
        self.composed_frame = tk.Frame(self)
        self.composed_frame.pack(pady=(10, 2))

        # Add buttons to test play the composed level
        self.play_composed_button = tk.Button(self.composed_frame, text="Play Composed Level", command=self.play_composed_level)
        self.play_composed_button.pack(side=tk.LEFT, padx=2)
        self.astar_composed_button = tk.Button(self.composed_frame, text="Use A* on Composed Level", command=self.astar_composed_level, state=tk.DISABLED)
        self.astar_composed_button.pack(side=tk.LEFT, padx=2)

        # Checkbox for switching between original and SNES graphics
        self.use_snes_graphics = tk.BooleanVar(value=False)
        self.graphics_checkbox = ttk.Checkbutton(
            self.composed_frame,
            text="Use SNES Graphics",
            variable=self.use_snes_graphics,
            state=tk.DISABLED
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
        self.build_mm_level_button = tk.Button(
            self.composed_frame,
            text="Build Mega Man Level",
            command=self.open_megaman_layout_editor,
            state=tk.DISABLED
        )
        self.build_mm_level_button.pack(side=tk.LEFT, padx=2)

        self.large_view_button = tk.Button(
            self.composed_frame,
            text="Large View",
            command=self.show_large_composed_view
        )
        self.large_view_button.pack(side=tk.LEFT, padx=2)

        # Thumbnails for composed level
        self.composed_thumb_frame = tk.Frame(self)
        self.composed_thumb_frame.pack(fill=tk.X)


        
        # Game selection
        # Mapping from the game on the display, to the actual internal names of each game
        self.game_display_to_real_mapping = {
            "Mario": "Mario",
            "Lode Runner": "LR",
            "Mega Man (Simple)": "MM-Simple",
            "Mega Man (Full)": "MM-Full",
            "Mega Man (Maker)": "MMLV",
            "Mario Maker 2": "MM2"
        }

        #Method called every time the dropdown is updated to use the mapping, and putting it in self.game
        def on_game_select(Event=None):
            game_display_var = self.game_display_var.get()
            self.game.set(self.game_display_to_real_mapping.get(game_display_var, game_display_var))

            # Auto-switch tileset AND dataset to match the newly selected game, so we don't
            # end up applying e.g. the MM-Simple tileset to an MM-Full dataset (and vice versa).
            game_to_tileset = {
                "Mario": common_settings.MARIO_TILESET,
                "LR": common_settings.LR_TILESET,
                "MM-Simple": common_settings.MM_SIMPLE_TILESET,
                "MM-Full": common_settings.MM_FULL_TILESET,
                "MMLV": common_settings.MMLV_TILESET,
                "MM2": common_settings.MM2_TILESET
            }

            new_tileset_path = game_to_tileset.get(self.game.get())

            tileset_changed = (new_tileset_path and os.path.isfile(new_tileset_path)
                                and new_tileset_path != self.tileset_path)

            if tileset_changed:
                self.tileset_path = new_tileset_path

            self._update_game_specific_controls()

            if tileset_changed and self.dataset_path and self.tileset_path:
                self.load_files_from_paths(self.dataset_path, self.tileset_path)

        #Creating the game dropdown
        self.game_display_var = tk.StringVar(value="Mario")
        self.game = tk.StringVar(value=self.game_display_to_real_mapping[self.game_display_var.get()])
        self.game_label = ttk.Label(self.composed_frame, text="Select Game:", style="TLabel")
        self.game_label.pack()
        self.game_dropdown = ttk.Combobox(self.composed_frame, textvariable=self.game_display_var, values=["Mario", "Lode Runner", "Mega Man (Simple)", "Mega Man (Full)", "Mega Man (Maker)", "Mario Maker 2"], state="readonly")
        self.game_dropdown.pack()
        self.game_dropdown.bind("<<ComboboxSelected>>", on_game_select)
        self._update_game_specific_controls()


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
        self.bind("<Up>", lambda e: self.prev_caption())
        self.bind("<Down>", lambda e: self.next_caption())

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
        #print("TILE DESCRIPTORS =", self.tile_descriptors)

        self.dataset_path = dataset_path
        try:
            with open(dataset_path, 'r', encoding='utf-8') as f:
                self.dataset = json.load(f)

            # Jacob: There is code from MarioDiffusion and MarioMakerPCG below,
            #        but I don't think that either approach is up to date.

            # Jacob: This code is from MarioDiffusion
            # Is designed to typically expect both scenes and captions, but if there are only level scenes,
            # convert the data format
            if isinstance(self.dataset, list) and all(isinstance(item, list) for item in self.dataset):
                # Convert to dict format with empty caption
                self.dataset = [{'scene': item, 'caption': ''} for item in self.dataset]

            # Jacob: This commented out code came from MarioMakerPCG
            # Normalize every sample to a dict with 'scene' and 'captions' keys.
            # Some datasets are raw scene grids (list of lists); others are
            # dicts missing 'caption'/'captions' (e.g. mm2pipeline_data.dataset's
            # {'name', 'scene'} output).
            #normalized_dataset = []
            #for item in self.dataset:
            #    if isinstance(item, list):
            #        normalized_dataset.append({'scene': item, 'captions': ['']})
            #    else:
            #        # MarioMaker_llm_captions.py stores multiple captions as
            #        # 'caption', 'caption1', 'caption2', ... ; collect them all
            #        # into an in-memory 'captions' list for display/navigation.
            #        captions = [item['caption']] if item.get('caption') else []
            #        idx = 1
            #        while f'caption{idx}' in item:
            #            captions.append(item[f'caption{idx}'])
            #            idx += 1
            #        if not captions:
            #            captions = ['']
            #        item['captions'] = captions
            #        item.setdefault('caption', captions[0])
            #        normalized_dataset.append(item)
            #self.dataset = normalized_dataset

            # Jacob: Neither of the two options above seem to be aware of the 
            #        new multi-caption approach

            _, self.id_to_char, self.char_to_id, self.tile_descriptors = extract_tileset(tileset_path)
            self.color_map = self._build_color_map()
            self.current_sample_idx = 0
            # Jacob: favor current_caption_idx instead of caption_cycle_idx
            self.current_caption_idx = 0
            # self.caption_cycle_idx = 0 
            self.redraw()
        except Exception as e:
            print(f"Error loading files: {e}")
            raise e

    # Jacob: Not sure this is necesary, or compatible with the other games.
    def _build_color_map(self):
        TAG_COLORS = [
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
        color_map = {}
        for tile_id, char in self.id_to_char.items():
            descriptors = self.tile_descriptors.get(char, set())
            color = (0.80, 0.80, 0.80)
            for tag, col in TAG_COLORS:
                if tag in descriptors:
                    color = col
                    break
            color_map[tile_id] = color
        return color_map

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

    # Jacob: I'm pretty sure this has never worked right and should be removed
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
            # Jacob: Not sure whether "caption" or "captions" is the right key to use here, but this is a fallback for datasets that don't have captions.
            sample = {"scene": sample, "caption": "No caption available."}
            # sample = {"scene": sample, "captions": ["No caption available."]}

        # Normalize captions: prefer explicit 'captions' list; otherwise collect
        # legacy caption fields (caption, caption1, caption2, ...) or fall back
        # to a single 'caption' string. Use `current_caption_idx` to index into
        # the resulting `captions` list consistently across the UI.
        if isinstance(sample, dict) and isinstance(sample.get('captions'), list):
            captions = sample['captions']
        else:
            caption_keys = self._sorted_caption_keys(
                [k for k in sample if isinstance(k, str) and k.startswith("caption")]
            )
            if caption_keys:
                captions = [sample.get(k, '') for k in caption_keys]
            else:
                captions = [sample.get('caption', '')]

        self.current_caption_idx = max(0, min(self.current_caption_idx, len(captions) - 1))
        self.caption_index_label.config(text=f"Caption {self.current_caption_idx + 1} / {len(captions)}")

        # Dynamically update tile and canvas size for this scene
        self.update_tile_and_canvas_size(sample['scene'])

        # Generate unique colors for caption phrases based on TOPIC_KEYWORDS
        from captions.caption_match import TOPIC_KEYWORDS # Mario
        from captions.LR_caption_match import TOPIC_KEYWORDS as LR_TOPIC_KEYWORDS
        from captions.MM_caption_match import TOPIC_KEYWORDS as MM_TOPIC_KEYWORDS
        # Jacob: TODO: Add Mario Maker

        # Generate a palette of distinct colors algorithmically
        # See if running Lode Runner
        if self.game.get()=="LR":
            TOPIC_KEYWORDS = LR_TOPIC_KEYWORDS
        elif self.game.get()=="MM-Simple" or self.game.get()=="MM-Full" or self.game.get()=="MMLV":
            TOPIC_KEYWORDS = MM_TOPIC_KEYWORDS
        # If not Lode Runner or Mega Man, use the default topic keywords of Mario
        
        # Jacob: Why was this next assignment statement ever here?
        #else:
        #    TOPIC_KEYWORDS = TOPIC_KEYWORDS
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

        

        if getattr(self, 'show_real', False) and self._render_real_image():
            # Real source image drawn onto the canvas; nothing else to draw.
            pass
        elif getattr(self, 'show_images', False):
            # Display as image using visualize_samples
            from level_dataset import visualize_samples
            import PIL.ImageTk
            from PIL import Image

            # With the A* overlay on, render the path-annotated image instead of the plain
            # one; fall back to the plain render if the path can't be produced.
            image = None
            if getattr(self, 'show_astar_path', False):
                image = self._astar_overlay_image(sample['scene'])

            if image is None:
                #Get the right size for the one-hot encoding
                if self.game.get()=="Mario":
                    num_classes = common_settings.MARIO_TILE_COUNT
                elif self.game.get()=="LR":
                    num_classes = common_settings.LR_TILE_COUNT
                elif self.game.get()=="MM-Simple":
                    num_classes = common_settings.MM_SIMPLE_TILE_COUNT
                elif self.game.get()=="MMLV":
                    num_classes = common_settings.MMLV_TILE_COUNT
                elif self.game.get()=="MM-Full":
                    num_classes = common_settings.MM_FULL_TILE_COUNT
                elif self.game.get()=="MM2":
                    num_classes = common_settings.MM2_TILE_COUNT
                else: # Jacob: MarioMakerPCG used this case, 
                      #        but I wonder if this should be an exception instead.
                    #Get the right size for the one-hot encoding
                    num_classes = len(self.id_to_char)


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
            # Jacob: General colors used by MarioDiffusion originally.
            #        There may not be enough colors to accommodate MarioMaker,
            #        and the colors are not tailored to that game
            colors = level_dataset.colors()
            # Jacob: This is the more specific option from MarioMakerPCG,
            #        but I would prefer to remove it for a general solution.
            color_map = getattr(self, 'color_map', None) or {}
            base_colors = level_dataset.colors()

            HEIGHT = len(sample['scene'])
            WIDTH = len(sample['scene'][0])
            for y in range(HEIGHT):
                for x in range(WIDTH):
                    tile_id = sample['scene'][y][x]
                    text = str(tile_id) if self.show_ids.get() else self.id_to_char.get(tile_id, '?')
                    if self.game.get() == "MM2": # Jacob: From MarioMakerPCG
                        if tile_id in color_map:
                            r, g, b = color_map[tile_id]
                        elif tile_id < len(base_colors):
                            r, g, b = base_colors[tile_id]
                        else:
                            r, g, b = (0.80, 0.80, 0.80)
                    else: # Jacob: Default MarioDiffusion behavior
                        # Convert (r, g, b) float tuple to hex color string
                        r, g, b = colors[tile_id % len(colors)]
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

        # Refresh the filter_reason line for this entry (shown only while the toggle is on).
        self._update_filter_reason_display(sample)
        self._update_mmlv_button(sample)

        # Update the text box below the scene. It normally shows the scene's caption(s),
        # but can be toggled to show an optional 'prompt' field when one is present.
        has_prompt = isinstance(sample, dict) and 'prompt' in sample
        self._update_prompt_toggle_control(has_prompt)

        # Jacob: This first if-case is all from MarioDiffusion
        if self.show_prompt and has_prompt:
            # Showing the raw prompt: hide the caption-cycle controls and render it as
            # plain (uncolored) text, since prompts don't carry per-phrase detail coords.
            self._update_caption_cycle_controls(0)
            self.caption_text.configure(state="normal")
            self.caption_text.delete("1.0", tk.END)
            self.caption_text.tag_configure("black", foreground="black")
            self.caption_text.insert(tk.END, str(sample.get('prompt', '')), ("black", "center"))
        else:
            # Jacob: The start of this section is from MarioDiffusion

            # Show the currently selected caption from the normalized list.
            self._update_caption_cycle_controls(len(captions))

            self.caption_text.configure(state="normal")
            self.caption_text.delete("1.0", tk.END)
            caption_text = captions[self.current_caption_idx]
            caption_parts = caption_text.split('.')
            for part in caption_parts:
                part = part.strip()
                if part:
                    part = part + "."  # Add back period
                    color = phrase_colors.get(part, "black")  # Remove period for color lookup
                    part = part + " " # Add space for readability
                    self.caption_text.tag_configure(color, foreground=color)
                    self.caption_text.insert(tk.END, part, (color, "center"))
        # Grow the text box to fit the full content so long captions/prompts aren't clipped.
        self._resize_caption_box()
        # Do not set state to disabled, so user can select/copy
        # self.caption_text.configure(state="disabled")

        self.sample_label.config(
            text=f"Sample: {self.current_sample_idx + 1} / {len(self.dataset)}"
        )
        self.title(f"Tile Dataset Viewer - Sample {self.current_sample_idx + 1} / {len(self.dataset)}")

    def _sorted_caption_keys(self, keys):
        """Order caption fields so bare 'caption' comes first, then caption1, caption2, ...
        Any non-numeric suffix falls back to lexical order at the end."""
        def sort_key(k):
            suffix = k[len("caption"):]
            if suffix == "":
                return (0, 0, "")
            if suffix.isdigit():
                return (1, int(suffix), "")
            return (2, 0, suffix)
        return sorted(keys, key=sort_key)

    def _update_caption_cycle_controls(self, num_captions):
        """Show the caption cycle button only when the current scene has more than one
        caption field; otherwise hide it so the browser looks/behaves exactly as before."""
        if num_captions > 1:
            if not self.caption_cycle_button.winfo_ismapped():
                self.caption_cycle_button.pack(side=tk.LEFT, padx=5)
                self.caption_cycle_label.pack(side=tk.LEFT, padx=5)
            self.caption_cycle_label.config(
                text=f"Caption {self.current_caption_idx + 1} / {num_captions}"
            )
        else:
            self.caption_cycle_button.pack_forget()
            self.caption_cycle_label.pack_forget()

    def _update_game_specific_controls(self):
        """Enable game-specific UI controls based on the selected game."""
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

        self.astar_composed_button.config(state=tk.NORMAL if is_mario else tk.DISABLED)

        self.build_mm_level_button.config(state=tk.NORMAL if is_megaman else tk.DISABLED)

    def _update_prompt_toggle_control(self, has_prompt):
        """Show the prompt toggle button only when the current scene has a 'prompt' field;
        otherwise hide it. The label reflects what the button will switch the box to."""
        if has_prompt:
            if not self.prompt_toggle_button.winfo_ismapped():
                self.prompt_toggle_button.pack(side=tk.LEFT, padx=5)
            self.prompt_toggle_button.config(
                text="Show Caption" if self.show_prompt else "Show Prompt"
            )
        else:
            self.prompt_toggle_button.pack_forget()

    def toggle_prompt(self):
        """Toggle the text box below the scene between its caption and its 'prompt' field."""
        self.show_prompt = not self.show_prompt
        self.redraw()

    def _resize_caption_box(self):
        """Grow/shrink the caption box so the whole caption is visible without scrolling.
        Uses the widget's wrapped display-line count; no-ops until the widget is laid out
        (e.g. before the window is first mapped), where geometry isn't known yet."""
        self.caption_text.update_idletasks()
        if not self.caption_text.winfo_ismapped():
            # Not laid out yet (e.g. the CLI-loaded first sample, before the window is
            # mapped). Retry once the event loop is idle and geometry is known.
            self.caption_text.after(50, self._resize_caption_box)
            return  # geometry not established yet; keep the default height for now
        try:
            lines = self.caption_text.count("1.0", "end-1c", "displaylines")
        except tk.TclError:
            return
        if not lines:
            return
        n = lines[0] if isinstance(lines, (tuple, list)) else lines
        self.caption_text.configure(height=max(3, int(n)))

    def cycle_caption(self):
        """Advance to the next caption field on the current scene and redraw."""
        if not self.dataset:
            return
        sample = self.dataset[self.current_sample_idx]
        if not isinstance(sample, dict):
            return
        # Normalize to the same captions list used by redraw()
        if isinstance(sample.get('captions'), list):
            captions = sample['captions']
        else:
            caption_keys = self._sorted_caption_keys(
                [k for k in sample if isinstance(k, str) and k.startswith("caption")]
            )
            captions = [sample.get(k, '') for k in caption_keys] if caption_keys else [sample.get('caption', '')]
        if len(captions) <= 1:
            return
        self.current_caption_idx = (self.current_caption_idx + 1) % len(captions)
        self.redraw()

    def prev_sample(self):
        if self.current_sample_idx > 0:
            self.current_sample_idx -= 1
            self.current_caption_idx = 0
            #self.caption_cycle_idx = 0
            self.redraw()

    def next_sample(self):
        if self.current_sample_idx < len(self.dataset) - 1:
            self.current_sample_idx += 1
            self.current_caption_idx = 0
            #self.caption_cycle_idx = 0
            self.redraw()

    def jump_to_sample(self, event=None):
        try:
            idx = int(self.jump_entry.get()) - 1
            if 0 <= idx < len(self.dataset):
                self.current_sample_idx = idx
                self.current_caption_idx = 0
                #self.caption_cycle_idx = 0
                self.redraw()
            else:
                print("Index out of range.")
        except ValueError:
            print("Invalid index entered.")

    def prev_caption(self):
        if not self.dataset:
            return
        captions = self.dataset[self.current_sample_idx].get('captions') or ['']
        if self.current_caption_idx > 0:
            self.current_caption_idx -= 1
            self.redraw()

    def next_caption(self):
        if not self.dataset:
            return
        captions = self.dataset[self.current_sample_idx].get('captions') or ['']
        if self.current_caption_idx < len(captions) - 1:
            self.current_caption_idx += 1
            self.redraw()

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

    def show_large_composed_view(self):
        """Pop up a large rendering of the full composed level, optionally with the
        A* path overlaid if 'Toggle A* Path' is currently on."""
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
        """Show a (possibly large) PIL image in a scrollable popup window."""
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
        canvas._photo_ref = photo  # keep a ref so it isn't garbage-collected
        canvas.create_image(0, 0, image=photo, anchor="nw")
        canvas.configure(scrollregion=(0, 0, pil_img.width, pil_img.height))
        win.geometry(f"{min(pil_img.width + 24, 1200)}x{min(pil_img.height + 24, 800)}")

    def play_composed_level(self):
        scene = self.merge_selected_scenes()
        if scene:
            level = self.get_sample_output(scene, use_snes_graphics=self.use_snes_graphics.get())
            if self.game.get() =="LR" and not self.validate_lode_runner_level(scene):
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
            # Jacob: Seems odd that there is a special case for MM2 here.
            if self.game.get() == "MM2":
                # No Java sim for MM2, use the Python astar/ check instead
                from astar.astar_traversability_check import astar_console_report
                print(astar_console_report(scene, id_to_char=self.id_to_char,
                                           tile_descriptors=self.tile_descriptors))
                return
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
        if self.game.get() == 'LR':
            char_grid = scene_to_ascii(scene, self.id_to_char, shorten=False)
            level = SampleOutput(level=scene, use_snes_graphics=use_snes_graphics)
        elif self.game.get() == 'Mario':
            # Mario
            if use_snes_graphics is None:
                use_snes_graphics = self.use_snes_graphics.get()
            char_grid = scene_to_ascii(scene, self.id_to_char)
            level = SampleOutput(level=char_grid, use_snes_graphics=use_snes_graphics)
        else: # Jacob: Mega Man also fails here I think
            # MM2 has no Java sim (it uses the Python A* path)
            raise ValueError(f"get_sample_output: no simulator for game {self.game.get()!r}")
        return level

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
    if len(sys.argv) >= 2:
        dataset_path = sys.argv[1]

    tileset_path = sys.argv[2] if len(sys.argv) == 3 else common_settings.MARIO_TILESET
        
    if dataset_path and not os.path.isfile(dataset_path):
        print("Invalid dataset path provided.")
        dataset_path = None

    if tileset_path and not os.path.isfile(tileset_path):
        print("Invalid tileset path provided.")
        tileset_path = None

    # Debugging
    #print("dataset_path", dataset_path)
    #print("tileset_path", tileset_path)
    app = TileViewer(dataset_path, tileset_path)
    app.protocol("WM_DELETE_WINDOW", app.on_close)
    app.mainloop()