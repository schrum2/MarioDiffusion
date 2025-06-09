import tkinter as tk
from PIL import Image, ImageTk, PngImagePlugin
from math import ceil, sqrt
import re
from util.sampler import SampleOutput
import os
from tkinter import filedialog
from util.sampler import scene_to_ascii

"""
Handles evolution in the latent space for generating level scenes.
"""

class ImageGridViewer:
    def __init__(self, root, callback_fn=None, back_fn=None, generation_fn = None, allow_prompt = False, allow_negative_prompt = False):
        self.root = root
        self.root.title("Generated Images")
        self.images = []  # Stores PIL Image objects
        self.genomes = []
        self.photo_images = []  # Stores PhotoImage objects (needed to prevent garbage collection)
        self.bottom_thumbnails = []  # Prevent GC for bottom frame thumbnails
        self.selected_images = set()  # Tracks which images are selected
        self.buttons = []  # Stores the button widgets
        self.callback_fn = callback_fn
        self.back_fn = back_fn
        self.generation_fn = generation_fn # get current generation number
        self.expanded_view = False  # Tracks if an image is currently expanded
        self.expanded_image_idx = None  # Tracks which image is expanded
        # For tracking composed scenes and thumbnails
        self.composed_scenes = []
        self.composed_thumbnails = []
        self.composed_thumbnail_labels = []
        self.selected_composed_index = None
        
        self.id_to_char = None # Will come later

        # Initial window sizing
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        
        # Set initial window size to 75% of screen
        window_width = int(screen_width * 0.75)
        window_height = int(screen_height * 0.75)
        root.geometry(f"{window_width}x{window_height}")


        self.main_container = tk.Frame(self.root)
        self.main_container.pack(expand=True, fill=tk.BOTH)

        self.main_container.rowconfigure(0, weight=1)  # Image grid gets priority
        self.main_container.rowconfigure(1, weight=0)  # Bottom frame is fixed
        self.main_container.columnconfigure(0, weight=1)

        # Image grid
        self.image_frame = tk.Frame(self.main_container)
        self.image_frame.grid(row=0, column=0, sticky="nsew", pady=(10, 5))

        # Constructed level frame
        self.bottom_frame = tk.Frame(self.main_container, height=80, bg="lightgrey")
        self.bottom_frame.grid(row=1, column=0, sticky="ew", padx=10, pady=5)
        self.bottom_frame.grid_propagate(False)

        
        # Create frame for control buttons and text inputs
        self.control_frame = tk.Frame(self.main_container, height=60)
        self.control_frame.grid(row=2, column=0, sticky="ew", padx=10, pady=5)
        self.control_frame.grid_propagate(False)

        #self.control_frame.pack_propagate(False)  # Prevent frame from shrinking
        
        # Create button frame
        self.button_frame = tk.Frame(self.control_frame)
        self.button_frame.pack(fill=tk.X)
        
        self.main_container.rowconfigure(2, weight=0)  # Ensure control frame doesn't expand

        # Add Back button
        self.back_button = tk.Button(
            self.button_frame,
            text="Previous Generation",
            command=self._handle_back,
            width=20,
            state=tk.DISABLED  # Initially disabled
        )
        self.back_button.pack(side=tk.LEFT, padx=5, pady=5)

        # Add Done button
        self.done_button = tk.Button(
            self.button_frame,
            text="Initialize",
            command=self._handle_done,
            width=20
        )
        self.done_button.pack(side=tk.LEFT, padx=5, pady=5)

        # Add Close button
        self.close_button = tk.Button(
            self.button_frame,
            text="Close",
            command=self.root.destroy,
            width=20
        )
        self.close_button.pack(side=tk.LEFT, padx=5, pady=5)
        
        self.play_composed_button = tk.Button(
            self.button_frame,
            text="Play Composed Level",
            command=self._play_composed_level,
            width=20
        )
        self.play_composed_button.pack(side=tk.LEFT, padx=5, pady=5)

        self.astar_composed_button = tk.Button(
            self.button_frame,
            text="A* Composed Level",
            command=self._astar_composed_level,
            width=20
        )
        self.astar_composed_button.pack(side=tk.LEFT, padx=5, pady=5)

        self.save_composed_button = tk.Button(
            self.button_frame,
            text="Save Composed Level",
            command=self._save_composed_level,
            width=20
        )
        self.save_composed_button.pack(side=tk.LEFT, padx=5, pady=5)

        self.clear_composed_button = tk.Button(
            self.button_frame,
            text="Clear Composed Level",
            command=self._clear_composed_level,
            width=20
        )
        self.clear_composed_button.pack(side=tk.LEFT, padx=5, pady=5)

        # Add these after your other control buttons (e.g., after self.clear_composed_button)
        self.delete_scene_button = tk.Button(
            self.button_frame,
            text="Delete Scene",
            command=self.delete_selected_composed_scene,
            width=15
        )
        self.delete_scene_button.pack(side=tk.LEFT, padx=5, pady=5)

        self.move_left_button = tk.Button(
            self.button_frame,
            text="Move Left",
            command=lambda: self.move_selected_composed_scene(-1),
            width=12
        )
        self.move_left_button.pack(side=tk.LEFT, padx=5, pady=5)

        self.move_right_button = tk.Button(
            self.button_frame,
            text="Move Right",
            command=lambda: self.move_selected_composed_scene(1),
            width=12
        )
        self.move_right_button.pack(side=tk.LEFT, padx=5, pady=5)

        # toggle checkbox for SNES graphics
        self.use_snes_graphics = tk.BooleanVar(value=False)
        self.snes_checkbox = tk.Checkbutton(
            self.control_frame,
            text="Use SNES Graphics",
            variable=self.use_snes_graphics
        )
        self.snes_checkbox.pack(side=tk.LEFT, padx=5, pady=5)
        
        self.allow_prompt = allow_prompt
        self.allow_negative_prompt = allow_negative_prompt
        self.negative_prompt_var = None
        self.negative_prompt_entry = None
        if self.allow_prompt:
            # Prompt input at the bottom (for conditional models)
            self.prompt_var = tk.StringVar()
            prompt_label = tk.Label(self.control_frame, text="Prompt:")
            prompt_label.pack(side=tk.LEFT, padx=(10, 2))
            self.prompt_entry = tk.Entry(self.control_frame, textvariable=self.prompt_var, width=60)
            self.prompt_entry.pack(side=tk.LEFT, padx=(0, 10))
            if self.allow_negative_prompt:
                self.negative_prompt_var = tk.StringVar()
                neg_label = tk.Label(self.control_frame, text="Negative Prompt:")
                neg_label.pack(side=tk.LEFT, padx=(10, 2))
                self.negative_prompt_entry = tk.Entry(self.control_frame, textvariable=self.negative_prompt_var, width=60)
                self.negative_prompt_entry.pack(side=tk.LEFT, padx=(0, 10))

    def set_negative_prompt_supported(self, supported: bool):
        self.negative_prompt_supported = supported
        if supported and self.negative_prompt_entry is not None:
            neg_label = tk.Label(self.control_frame, text="Negative Prompt:")
            neg_label.pack(side=tk.LEFT, padx=(10, 2))
            self.negative_prompt_entry.pack(side=tk.LEFT, padx=(0, 10))
        elif not supported and self.negative_prompt_entry is not None:
            self.negative_prompt_entry.pack_forget()

        # Bind resize event
        self.root.bind('<Configure>', self._on_window_resize)

    def _clear_composed_level(self):
        self.composed_scenes.clear()
        self.composed_thumbnails.clear()
        for label in self.composed_thumbnail_labels:
            label.destroy()
        self.composed_thumbnail_labels.clear()
        self.selected_composed_index = None

    def delete_selected_composed_scene(self):
        idx = self.selected_composed_index
        if idx is not None and 0 <= idx < len(self.composed_scenes):
            # Remove from all lists
            self.composed_scenes.pop(idx)
            self.composed_thumbnails.pop(idx)
            label = self.composed_thumbnail_labels.pop(idx)
            label.destroy()
            self.selected_composed_index = None
            self.rebind_composed_thumbnail_clicks()
        else:
            tk.messagebox.showinfo("No selection", "Please select a scene to delete.")

    def move_selected_composed_scene(self, direction):
        idx = self.selected_composed_index
        if idx is None or not (0 <= idx < len(self.composed_scenes)):
            tk.messagebox.showinfo("No selection", "Please select a scene to move.")
            return

        new_idx = idx + direction
        if not (0 <= new_idx < len(self.composed_scenes)):
            return  # Out of bounds

        # Swap in all lists
        for lst in [self.composed_scenes, self.composed_thumbnails, self.composed_thumbnail_labels]:
            lst[idx], lst[new_idx] = lst[new_idx], lst[idx]

        # Re-pack labels in new order
        for lbl in self.composed_thumbnail_labels:
            lbl.pack_forget()
        for lbl in self.composed_thumbnail_labels:
            lbl.pack(side=tk.LEFT, padx=2)

        self.rebind_composed_thumbnail_clicks()
        self.select_composed_thumbnail(new_idx)

    def _save_composed_level(self):
        if self.composed_scenes:
            level = self.get_sample_output(self._merge_composed_scenes())
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
                level.save(file_path)
                print(f"Composed level saved to {file_path}")
            else:
                print("Save operation cancelled.")

    def _merge_composed_scenes(self):
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

    def get_sample_output(self, scene, use_snes_graphics=None):
        if use_snes_graphics is None:
            use_snes_graphics = self.use_snes_graphics.get()
        char_grid = scene_to_ascii(scene, self.id_to_char, False)
        return SampleOutput(level=char_grid, use_snes_graphics=use_snes_graphics)

    def _play_composed_level(self):
        composed_scene = self._merge_composed_scenes()
        if composed_scene:
            level = self.get_sample_output(composed_scene, use_snes_graphics=self.use_snes_graphics.get())
            level.play()

    def _astar_composed_level(self):
        composed_scene = self._merge_composed_scenes()
        if composed_scene:
            level = self.get_sample_output(composed_scene, use_snes_graphics=self.use_snes_graphics.get())
            console_output = level.run_astar()
            print(console_output)

    def get_available_scenes(self):
        """Returns a list of available scenes from the genomes."""
        return [g.scene for g in self.genomes if g.scene is not None]

    def clear_images(self):
        """Clears all images from the grid and resets selections."""
        self.images.clear()
        self.genomes.clear()
        self.selected_images.clear()
        self.expanded_view = False
        self.expanded_image_idx = None
        self._update_grid()

    def add_image(self, pil_image, genome=None):
        """
        Add a new image to the grid.

        """
        self.images.append(pil_image)
        self.genomes.append(genome)
        self._update_grid()
        
    def get_selected_images(self):
        """Returns list of selected PIL Image objects."""
        return [(i,self.images[i]) for i in self.selected_images]
    
    def _calculate_thumbnail_size(self):
        """Calculate thumbnail size based on current window dimensions."""
        # Get current window size
        window_width = self.root.winfo_width()
        window_height = self.root.winfo_height() - 120  # Adjusted for larger control frame
        
        # Calculate grid dimensions for 3x3 grid
        n_images = len(self.images)
        if n_images == 0:
            return (256, 256)  # Default size if no images
        
        if self.expanded_view:
            # For expanded view, use most of the available space
            return (window_width - 20, window_height - 20)
        
        grid_size = min(3, ceil(sqrt(n_images)))
        
        # Calculate thumbnail size to fit the grid with some padding
        padding = 50  # Additional padding for margins and buttons
        max_thumb_width = (window_width - (grid_size + 1) * 10) // grid_size
        max_thumb_height = (window_height - (grid_size + 1) * 10 - padding) // grid_size
        
        # Ensure thumbnail has equal width and height
        button_height = 40  # Estimated height for buttons below image
        max_thumb_height -= button_height
        thumbnail_size = min(max_thumb_width, max_thumb_height)
        
        return (thumbnail_size, thumbnail_size)
    
    def _create_tooltip(self, widget, text):
        """Create a tooltip for a widget."""
        def enter(event):
            # Create a toplevel window
            tooltip = tk.Toplevel()
            tooltip.wm_overrideredirect(True)  # Remove window decorations
            
            # Position tooltip near the mouse
            x, y, _, _ = widget.bbox("insert")
            x += widget.winfo_rootx() + 25
            y += widget.winfo_rooty() + 20
            
            # Create tooltip label
            label = tk.Label(tooltip, text=text, justify=tk.LEFT,
                           background="#ffffe0", relief=tk.SOLID, borderwidth=1)
            label.pack()
            
            tooltip.wm_geometry(f"+{x}+{y}")
            widget._tooltip = tooltip
            
        def leave(event):
            # Destroy tooltip when mouse leaves
            if hasattr(widget, '_tooltip'):
                widget._tooltip.destroy()
                del widget._tooltip
        
        def check_mouse(event):
            if not (0 <= event.x <= widget.winfo_width() and 0 <= event.y <= widget.winfo_height()):
                leave(event)

        if text:
            widget.bind('<Enter>', enter)
            widget.bind('<Leave>', leave)
            widget.bind('<Motion>', check_mouse)
            widget.bind('<ButtonPress>', leave)
            widget.bind('<FocusOut>', leave)
            widget.bind('<Unmap>', leave)
            widget.bind('<Destroy>', leave)
    
    def _on_window_resize(self, event):
        """Handles window resize event."""
        # Only update if the resize is significant to prevent excessive redraws
        if event.widget == self.root:
            self._update_grid()
    
    def _toggle_expanded_view(self, idx):
        """Toggle expanded view for an image."""
        if self.expanded_view and self.expanded_image_idx == idx:
            # Already expanded this image, return to grid view
            self.expanded_view = False
            self.expanded_image_idx = None
        else:
            # Expand this image
            self.expanded_view = True
            self.expanded_image_idx = idx
        
        self._update_grid()
    
    def _update_grid(self):
        # Clear existing buttons
        for button in self.buttons:
            button.destroy()
        self.buttons.clear()
        self.photo_images.clear()
        
        # Calculate grid dimensions
        n_images = len(self.images)
        if n_images == 0:
            return
        
        if self.expanded_view:
            # Show only the expanded image
            idx = self.expanded_image_idx
            img = self.images[idx]
            
            # Get dynamic size for expanded view
            expanded_size = self._calculate_thumbnail_size()
            
            # Create a copy and resize for display
            display_img = img.copy()
            
            # Calculate aspect ratio of the original image
            original_width, original_height = img.size
            aspect_ratio = original_width / original_height
            
            # Calculate new dimensions while preserving aspect ratio
            expanded_width, expanded_height = expanded_size
            if aspect_ratio > 1:  # Wider than tall
                new_width = expanded_width
                new_height = int(expanded_width / aspect_ratio)
                if new_height > expanded_height:
                    new_height = expanded_height
                    new_width = int(expanded_height * aspect_ratio)
            else:  # Taller than wide or square
                new_height = expanded_height
                new_width = int(expanded_height * aspect_ratio)
                if new_width > expanded_width:
                    new_width = expanded_width
                    new_height = int(expanded_width / aspect_ratio)
            
            # Resize the image
            display_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # Convert to PhotoImage
            photo = ImageTk.PhotoImage(display_img)
            self.photo_images.append(photo)
            
            # Create button with expanded image
            btn = tk.Button(
                self.image_frame,
                image=photo,
                relief='solid',
                borderwidth=2,
                command=lambda i=idx: self._toggle_expanded_view(i)
            )
            
            # Add tooltip
            self._create_tooltip(btn, self.genomes[idx].__str__())
            
            # Position in grid
            btn.grid(row=0, column=0, padx=5, pady=5, sticky='nsew')
            
            # Configure grid weights for center alignment
            self.image_frame.grid_rowconfigure(0, weight=1)
            self.image_frame.grid_columnconfigure(0, weight=1)
            
            self.buttons.append(btn)
            
            # Update selected state if necessary
            if idx in self.selected_images:
                btn.configure(bg='blue')
                
            # Add a label with exit instructions
            exit_label = tk.Label(
                self.image_frame,
                text="Click image to return to grid view",
                font=("Helvetica", 10),
                bg="light grey"
            )
            exit_label.grid(row=1, column=0, sticky='ew')
            self.buttons.append(exit_label)  # Add to buttons list so it gets cleaned up
            
        else:
            # Show normal grid
            # Dynamically calculate grid size
            grid_size = min(3, ceil(sqrt(n_images)))
            
            # Get dynamic thumbnail size
            thumbnail_size = self._calculate_thumbnail_size()
            thumbnail_size = (max(100,thumbnail_size[0]), max(100,thumbnail_size[1]))

            for idx, img in enumerate(self.images):
                # Create a copy and resize for thumbnail
                thumb = img.copy()
                thumb.thumbnail(thumbnail_size, Image.Resampling.LANCZOS)
                
                # Convert to PhotoImage
                photo = ImageTk.PhotoImage(thumb)
                self.photo_images.append(photo)
                
                # Create a container frame for image + buttons
                frame = tk.Frame(self.image_frame)

                # Image button
                btn = tk.Button(
                    frame,
                    image=photo,
                    relief='solid',
                    borderwidth=2
                )
                btn.pack()

                # Tooltip for image
                self._create_tooltip(btn, self.genomes[idx].__str__())

                # Selection behavior
                btn.configure(
                    command=lambda i=idx, b=btn: self._toggle_selection(i, b)
                )

                # Double-click to expand
                btn.bind('<Double-Button-1>', lambda event, i=idx: self._toggle_expanded_view(i))

                # Button container for horizontal layout
                button_row = tk.Frame(frame)
                button_row.pack(pady=(2, 2))

                # "Play" button
                play_button = tk.Button(
                    button_row,
                    text="Play",
                    command=lambda g=self.genomes[idx]: self._play_genome(g)
                )
                play_button.pack(side='left', padx=(0, 5))

                # "A* Agent" button
                astar_button = tk.Button(
                    button_row,
                    text="A* Agent",
                    command=lambda g=self.genomes[idx]: self._run_astar_agent(g)
                )
                astar_button.pack(side='left')

                # "Add To Level" button
                add_button = tk.Button(
                    button_row,
                    text="Add To Level",
                    command=lambda i=idx: self._add_to_level(i)
                )
                add_button.pack(side='left', padx=(5, 0))

                # Position in grid
                row = idx // grid_size
                col = idx % grid_size
                frame.grid(row=row, column=col, padx=5, pady=5, sticky='nsew')
                
                # Configure grid weights to make buttons resize
                self.image_frame.grid_rowconfigure(row, weight=1)
                self.image_frame.grid_columnconfigure(col, weight=1)
                
                self.buttons.append(frame)
                
                # Update selected state if necessary
                if idx in self.selected_images:
                    btn.configure(bg='blue')

    def select_composed_thumbnail(self, index):
        # Deselect all
        for lbl in self.composed_thumbnail_labels:
            lbl.config(relief="flat", borderwidth=2)
        # Select the clicked one
        self.composed_thumbnail_labels[index].config(relief="solid", borderwidth=3)
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

    def _add_to_level(self, idx):
        # Store a copy of the scene
        scene = self.genomes[idx].scene
        self.composed_scenes.append(scene)

        # Create and store the thumbnail
        img = self.images[idx].copy()
        img.thumbnail((64, 64), Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(img)
        self.composed_thumbnails.append(photo)  # Prevent GC

        # Create a clickable label for the thumbnail
        label = tk.Label(self.bottom_frame, image=photo, borderwidth=2, relief="flat")
        label.pack(side=tk.LEFT, padx=2)
        self.composed_thumbnail_labels.append(label)
        self.rebind_composed_thumbnail_clicks()

    def _play_genome(self, genome):
        # level = self.get_sample_output(genome.scene)
        level = self.get_sample_output(genome.scene, use_snes_graphics=self.use_snes_graphics.get())
        level.play()

    def _run_astar_agent(self, genome):
        # level = self.get_sample_output(genome.scene)
        level = self.get_sample_output(genome.scene, use_snes_graphics=self.use_snes_graphics.get())
        console_output = level.run_astar()
        print(console_output)

    def _toggle_selection(self, idx, button):
        # Don't toggle selection if in expanded view
        if self.expanded_view:
            return
            
        if idx in self.selected_images:
            self.selected_images.remove(idx)
            button.configure(bg='SystemButtonFace')  # Default background
        else:
            self.selected_images.add(idx)
            button.configure(bg='blue')  # Highlight selected

        if len(self.selected_images) == 0:
            self.done_button.config(text="Reset")
        else:
            self.done_button.config(text="Evolve Selected")

    def _handle_done(self):
        """Called when Evolve button is clicked"""

        self.done_button.config(text="Reset")
        if self.callback_fn:
            selected = self.get_selected_images()
            param_values = dict()
            if self.allow_prompt:
                prompt = self.prompt_entry.get()
                if prompt != None and prompt.strip() != "": 
                    param_values["prompt"] = prompt
                if self.allow_negative_prompt:
                    negative_prompt = self.negative_prompt_entry.get()
                    if negative_prompt != None and negative_prompt.strip() != "":
                        param_values["negative_prompt"] = negative_prompt
                    
            self.callback_fn(selected, **param_values)

        self.update_back_button_status()

    def update_back_button_status(self):
        if self.generation_fn() > 0:
            # Can only go back if not at the start
            self.back_button.config(state=tk.NORMAL)
        else:
            self.back_button.config(state=tk.DISABLED)

    def _evolve_latents(self):
        selected = self.get_selected_images()
        for (i,image) in selected:
            self.genomes[i].store_latents_in_genome()
            #print(f"{i}: {self.genomes[i].__str__()}, {self.genomes[i].metadata()}")

        self._update_grid()
    
    def _save_selected(self):
        selected = self.get_selected_images()
        for (i,image) in selected:
            full_desc = self.genomes[i].__str__()
            image_meta = self.genomes[i].metadata()

            metadata = PngImagePlugin.PngInfo()
            for key in image_meta:
                metadata.add_text(f"sd_{key}", str(image_meta[key]))

            match = re.search(r"id=(\d+)", full_desc)
            output = f"Image_Id{match.group(1)}_Num{i}.png"
            image.save(output, "PNG", pnginfo=metadata)
            print(f"Saved {output}")

    def _handle_back(self):
        """Called when Back button is clicked"""

        if self.back_fn:
            self.back_fn()

        self.update_back_button_status()
