import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
import sys
from PIL import Image, ImageTk

from level_dataset import mario_tiles, lr_tiles, mm_tiles
from MM2_Files.render_mm2 import mm2_tiles
from util.sampler import scene_to_ascii

# Jacob: Despite the name, I think some of the standard LevelEditor class applies to all games.

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
        # Jacob: These two lines were in MarioMakerPCG, but I'm not sure they are needed
        # self.grid_frame = ttk.Frame(master)
        # self.grid_frame.pack(padx=10, pady=10)

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
        # self.tile_buttons = [] # Jacob: Why did MarioMakerPCG add this?
        self.tile_photo_images = []
        self.palette_photo_images = []

        self.cell_frames = {}
        self.cell_labels = {}

        for r, row in enumerate(self.scene):
            # button_row = [] # Jacob: Why did MarioMakerPCG add this?
            for c, tile_id in enumerate(row):
                frame = tk.Frame(
                    self.grid_frame,
                    highlightthickness=2,
                    highlightbackground=self.UNSELECTED_BORDER,
                    highlightcolor=self.UNSELECTED_BORDER,
                )
                frame.grid(row=r, column=c, padx=1, pady=1)

                photo = ImageTk.PhotoImage(self.tile_images[tile_id])
                # Jacob: this is code from MarioMakerPCG.
                #        I think the code from MarioDiffusion (below) is more up-to-date,
                #        so I commented the MarioMakerPCG code out
                #btn = ttk.Button(
                #    self.grid_frame,
                #    image=photo,
                #    command=lambda r=r, c=c: self.cycle_tile(r, c)
                #)
                #btn.image = photo
                #btn.grid(row=r, column=c, padx=1, pady=1)
                self.tile_photo_images.append(photo)
                label = tk.Label(frame, image=photo, borderwidth=0)
                label.image = photo
                label.pack()

                label.bind("<Button-1>", lambda e, r=r, c=c: self._left_click_cell(r, c, shift=bool(e.state & 0x0001)))
                label.bind("<Button-3>", lambda e, r=r, c=c: self._cycle_cell(r, c, -1))

                self.cell_frames[(r, c)] = frame
                self.cell_labels[(r, c)] = label
                
                # Jacob: Also from MarioMakerPCG
                #button_row.append(btn)
            # Jacob: Also from MarioMakerPCG
            #self.tile_buttons.append(button_row)

        # Palette tiles, arranged in a grid (side-by-side), in cycle order
        self.palette_swatch_frames = {}
        for tile_id in range(len(self.id_to_char)):
            self._add_palette_entry(palette_inner, tile_id)

        # From MarioDiffusion
        controls = ttk.Frame(outer)
        controls.pack(pady=(12, 0))
        ttk.Button(controls, text="Save", command=self.save, width=14).pack(side=tk.LEFT, padx=6)
        ttk.Button(controls, text="Cancel", command=master.destroy, width=14).pack(side=tk.LEFT, padx=6)
        # Jacob: From MarioMakerPCG
        #controls = ttk.Frame(master)
        #controls.pack(pady=8)
        #ttk.Button(controls, text="Save", command=self.save).pack(side=tk.LEFT, padx=4)
        #ttk.Button(controls, text="Cancel", command=master.destroy).pack(side=tk.LEFT, padx=4)
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

    # Jacob: From MarioMakerPCG. What is it for?
    #        It seems to have overlap with some code below, which makes me
    #        suspect it was replaced, but I'm not sure.
    def cycle_tile(self, row, col):
        current_id = self.scene[row][col]
        next_id = (current_id + 1) % len(self.id_to_char)
        self.scene[row][col] = next_id
        photo = ImageTk.PhotoImage(self.tile_images[next_id])
        btn = self.tile_buttons[row][col]
        btn.config(image=photo)
        btn.image = photo
        self.tile_photo_images.append(photo)
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
        # Accepts either CaptionBuilder's display names ("Mega Man (Full)") or the
        # internal game codes used by ascii_data_browser's TileViewer ("MM-Full"),
        # since both apps pass this class their own game string.
        if game in ("Lode Runner", "LR"):
            return lr_tiles()
        elif game in ("Mega Man (Simple)", "MM-Simple", "MM-simple"):
            return mm_tiles("MM-Simple")
        elif game in ("Mega Man (Full)", "MM-Full", "MM-full"):
            return mm_tiles("MM-Full")
        elif game in ("Mega Man (Maker)", "MMLV"):
            return mm_tiles("MMLV")
        elif game in ("Mario Maker 2", "MM2", "Mario Maker"):
            # MM2 per-tile sprites, indexed exactly like extract_tileset()/id_to_char
            # so tile id N in the scene maps to tile image N in the editor grid.
            return mm2_tiles()        
        else: # Mario case 
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

    Works with any "app" object that provides: composed_scenes, composed_thumbnails,
    _render_scene_image(scene), edit_composed_scene(idx, extra_on_save=None),
    _astar_path_for_scene(scene, spawn=None, orb=None), id_to_char, char_to_id.
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

    SEAM_THICKNESS = 10
    SEAM_SMOOTH_COLOR = "#33cc55"
    SEAM_LOCKED_COLOR = "#dd3333"
    SEAM_TILE_PX = 16
    SEAM_SCREEN_W_TILES = 16
    SEAM_SCREEN_H_TILES = 14

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

        # seam_locked: set of frozenset({(col1,row1), (col2,row2)}) for adjacent placed
        # cells the user wants a hard screen-lock at. Anything NOT in here is smooth —
        # that's the default.
        self.seam_locked = set()
        self.seam_canvas_ids = {}   # seam key -> canvas rect id

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

        # Floating legend explaining the seam-color strips, pinned to the bottom-left
        # corner of the grid view. Uses place() over canvas_frame so it stays fixed on
        # screen regardless of scrolling/zoom, instead of living on the scrollable canvas.
        legend = tk.Frame(canvas_frame, bg="#111111", bd=1, relief="solid")
        legend.place(in_=canvas_frame, relx=0.0, rely=1.0, x=10, y=-28, anchor="sw")

        smooth_row = tk.Frame(legend, bg="#111111")
        smooth_row.pack(anchor="w", padx=8, pady=(6, 2))
        smooth_swatch = tk.Frame(smooth_row, bg=self.SEAM_SMOOTH_COLOR, width=16, height=14)
        smooth_swatch.pack_propagate(False)
        smooth_swatch.pack(side=tk.LEFT, padx=(0, 6))
        tk.Label(smooth_row, text="Smooth scroll", bg="#111111", fg="white",
                 font=("Arial", 9)).pack(side=tk.LEFT)

        locked_row = tk.Frame(legend, bg="#111111")
        locked_row.pack(anchor="w", padx=8, pady=(0, 6))
        locked_swatch = tk.Frame(locked_row, bg=self.SEAM_LOCKED_COLOR, width=16, height=14)
        locked_swatch.pack_propagate(False)
        locked_swatch.pack(side=tk.LEFT, padx=(0, 6))
        tk.Label(locked_row, text="Screen lock", bg="#111111", fg="white",
                 font=("Arial", 9)).pack(side=tk.LEFT)

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

        self.seam_canvas_ids.clear()
        for cell in list(self.placements.keys()):
            self._draw_seams_for_cell(*cell)

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
        self._remove_seams_on_cell(col, row)
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
        self._draw_seams_for_cell(col, row)

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
        self._remove_seams_on_cell(col, row)
        scene_index = self.placements.pop((col, row), None)
        self._clear_cell_visual(col, row)
        if scene_index is not None:
            self.placed_scene_indices.discard(scene_index)
        self._populate_palette()

    def clear_grid(self):
        self.placements.clear()
        self.placed_scene_indices.clear()
        self.marker_placements.clear()
        self.seam_locked.clear()
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

    # ------------------------------------------------------------------ seams

    def _seam_rect_coords(self, cellA, cellB):
        (c1, r1), (c2, r2) = sorted([cellA, cellB])
        x1, y1 = self._cell_to_pixel(c1, r1)
        t = self.SEAM_THICKNESS / 2
        if c1 == c2:  # stacked vertically — seam is a horizontal strip between them
            seam_y = y1 + self.cell_pixels
            return (x1, seam_y - t, x1 + self.cell_pixels, seam_y + t)
        else:  # side by side — seam is a vertical strip between them
            seam_x = x1 + self.cell_pixels
            return (seam_x - t, y1, seam_x + t, y1 + self.cell_pixels)

    def _draw_seam(self, cellA, cellB):
        key = frozenset((cellA, cellB))
        old = self.seam_canvas_ids.pop(key, None)
        if old is not None:
            self.grid_canvas.delete(old)
        x0, y0, x1, y1 = self._seam_rect_coords(cellA, cellB)
        color = self.SEAM_LOCKED_COLOR if key in self.seam_locked else self.SEAM_SMOOTH_COLOR
        rect_id = self.grid_canvas.create_rectangle(x0, y0, x1, y1, fill=color, outline="")
        self.grid_canvas.tag_bind(rect_id, "<Button-1>", lambda e, k=key: self._toggle_seam(k))
        self.grid_canvas.tag_raise(rect_id)
        self.seam_canvas_ids[key] = rect_id

    def _toggle_seam(self, key):
        if key in self.seam_locked:
            self.seam_locked.discard(key)
        else:
            self.seam_locked.add(key)
        cellA, cellB = tuple(key)
        self._draw_seam(cellA, cellB)

    def _draw_seams_for_cell(self, col, row):
        for ncell in ((col + 1, row), (col - 1, row), (col, row + 1), (col, row - 1)):
            if ncell in self.placements:
                self._draw_seam((col, row), ncell)

    def _remove_seams_on_cell(self, col, row):
        for key in list(self.seam_canvas_ids.keys()):
            if (col, row) in key:
                self.grid_canvas.delete(self.seam_canvas_ids.pop(key))
                self.seam_locked.discard(key)

    def _seam_screen_pairs(self):
        """Expand self.seam_locked (layout-cell seams) into the set of adjacent
        screen-block coordinate pairs that vglc_to_mmlv.convert() should treat
        as hard-locked instead of smooth."""
        if not self.seam_locked:
            return set()
        cleaned_scenes, result = self._get_cleaned_scenes()
        if cleaned_scenes is None:
            return set()
        scene_h, scene_w = result
        min_col = min(c for c, r in self.placements)
        min_row = min(r for c, r in self.placements)

        SW, SH, PX = self.SEAM_SCREEN_W_TILES, self.SEAM_SCREEN_H_TILES, self.SEAM_TILE_PX

        pairs = set()
        for key in self.seam_locked:
            (colA, rowA), (colB, rowB) = tuple(key)
            if (colA, rowA) not in self.placements or (colB, rowB) not in self.placements:
                continue
            if rowA == rowB:
                left_col, right_col = sorted((colA, colB))
                boundary_tile_x = (right_col - min_col) * scene_w
                sx_left = ((boundary_tile_x - 1) // SW) * SW * PX
                sx_right = (boundary_tile_x // SW) * SW * PX
                row_tile_start = (rowA - min_row) * scene_h
                for local_row in range(0, scene_h, SH):
                    sy = ((row_tile_start + local_row) // SH) * SH * PX
                    pairs.add(((sx_left, sy), (sx_right, sy)))
            else:
                top_row, bottom_row = sorted((rowA, rowB))
                boundary_tile_y = (bottom_row - min_row) * scene_h
                sy_top = ((boundary_tile_y - 1) // SH) * SH * PX
                sy_bottom = (boundary_tile_y // SH) * SH * PX
                col_tile_start = (colA - min_col) * scene_w
                for local_col in range(0, scene_w, SW):
                    sx = ((col_tile_start + local_col) // SW) * SW * PX
                    pairs.add(((sx, sy_top), (sx, sy_bottom)))
        return pairs

    # ------------------------------------------------------------------ build merged scene

    def _get_cleaned_scenes(self):
            """Strip leading blank rows once; shared by build_merged_scene and
            _marker_grid_positions so they can never disagree on post-strip height."""
            blank_tid = self.app.char_to_id.get("@", 0)

            def strip_leading_blank_rows(scene):
                start = 0
                while start < len(scene) and all(tile == blank_tid for tile in scene[start]):
                    start += 1
                return scene[start:]

            scenes_raw = self.app.composed_scenes
            cleaned = {i: strip_leading_blank_rows(scenes_raw[i]) for i in self.placements.values()}
            dims = {(len(cleaned[i]), len(cleaned[i][0])) for i in self.placements.values()}
            if len(dims) > 1:
                return None, dims  # let caller report the mismatch
            scene_h, scene_w = next(iter(dims))
            return cleaned, (scene_h, scene_w)

    def _marker_grid_positions(self):
        """Absolute (col, row) in the merged grid for each placed marker key.

        Uses the same origin/scene geometry as the merge, so positions line up with
        build_merged_ascii's stamping and build_merged_scene's tile layout."""
        if not self.placements or not self.marker_placements:
            return {}
        cleaned_scenes, result = self._get_cleaned_scenes()
        if cleaned_scenes is None:
            return {}
        scene_h, scene_w = result
        min_col = min(c for c, r in self.placements)
        min_row = min(r for c, r in self.placements)

        positions = {}
        for key, (col, row, t_col, t_row) in self.marker_placements.items():
            scene_index = self.placements[(col, row)]
            raw_h = len(self.app.composed_scenes[scene_index])
            strip_amount = raw_h - len(cleaned_scenes[scene_index])
            adj_t_row = t_row - strip_amount
            if adj_t_row < 0:
                continue  # marker was on a row that got stripped away; drop it rather than misplace it
            positions[key] = ((col - min_col) * scene_w + t_col,
                               (row - min_row) * scene_h + adj_t_row)
        return positions

    def build_merged_scene(self):
        """Merge placed scenes into one tile-ID grid (no spawn/exit handling).
        Used for A* (with explicit or auto spawn/orb) and as the base for export."""
        if not self.placements:
            messagebox.showinfo("Empty layout", "Drag at least one scene onto the grid first.")
            return None

        cleaned_scenes, result = self._get_cleaned_scenes()
        if cleaned_scenes is None:
            messagebox.showerror(
                "Mismatched scene sizes",
                "All scenes must share the same width and height after removing blank rows.\n"
                "Found sizes (h×w): " + ", ".join(f"{h}×{w}" for h, w in result)
            )
            return None
        scene_h, scene_w = result
        blank_tid = self.app.char_to_id.get("@", 0)

        cols    = [c for c, r in self.placements]
        rows    = [r for c, r in self.placements]
        min_col = min(cols);  max_col = max(cols)
        min_row = min(rows);  max_row = max(rows)

        out_w  = (max_col - min_col + 1) * scene_w
        out_h  = (max_row - min_row + 1) * scene_h
        merged = [[blank_tid for _ in range(out_w)] for _ in range(out_h)]

        for (col, row), scene_index in self.placements.items():
            scene = cleaned_scenes[scene_index]
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
            locked_seams = self._seam_screen_pairs()
            result = convert(lines, level_name=level_name, author="AI", locked_seams=locked_seams)

            with open(mmlv_path, 'w', encoding='utf-8', newline='\n') as f:
                f.write(result)

            return True

        except Exception as e:
            messagebox.showerror("Save failed", str(e))
            return False