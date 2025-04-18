import tkinter as tk
from PIL import Image, ImageTk, PngImagePlugin
from math import ceil, sqrt
import io
import re
import json
from sampler import SampleOutput

"""
Handles evolution in the latent space for generating level scenes.
"""

class ImageGridViewer:
    def __init__(self, root, callback_fn=None, back_fn=None, generation_fn = None):
        self.root = root
        self.root.title("Generated Images")
        self.images = []  # Stores PIL Image objects
        self.genomes = []
        self.photo_images = []  # Stores PhotoImage objects (needed to prevent garbage collection)
        self.selected_images = set()  # Tracks which images are selected
        self.buttons = []  # Stores the button widgets
        self.callback_fn = callback_fn
        self.back_fn = back_fn
        self.generation_fn = generation_fn # get current generation number
        self.expanded_view = False  # Tracks if an image is currently expanded
        self.expanded_image_idx = None  # Tracks which image is expanded
        self.added_image_indexes = []
        
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
        self.bottom_frame = tk.Frame(self.main_container, height=150, bg="lightgrey")
        self.bottom_frame.grid(row=1, column=0, sticky="ew", padx=10, pady=5)
        self.bottom_frame.grid_propagate(False)

        
        # Create frame for control buttons and text inputs
        self.control_frame = tk.Frame(self.main_container, height=120)
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

        # Add Save button
        self.save_button = tk.Button(
            self.button_frame,
            text="Save Selected",
            command=self._save_selected,
            width=20
        )
        self.save_button.pack(side=tk.LEFT, padx=5, pady=5)

        # Add Close button
        self.close_button = tk.Button(
            self.button_frame,
            text="Close",
            command=self.root.destroy,
            width=20
        )
        self.close_button.pack(side=tk.LEFT, padx=5, pady=5)
        
        self.playall_button = tk.Button(
            self.button_frame,
            text="Combine And Play",
            command=self._play_all,
            width=20
        )
        self.playall_button.pack(side=tk.LEFT, padx=5, pady=5)

        self.astarall_button = tk.Button(
            self.button_frame,
            text="Combine And A*",
            command=self._run_astar_agent_all,
            width=20
        )
        self.astarall_button.pack(side=tk.LEFT, padx=5, pady=5)

        self.play_composed_button = tk.Button(
            self.button_frame,
            text="Play Composed Level",
            command=self._play_composed_level,
            width=20
        )
        self.play_composed_button.pack(side=tk.LEFT, padx=5, pady=5)

        self.clear_composed_button = tk.Button(
            self.button_frame,
            text="Clear Composed Level",
            command=self._clear_composed_level,
            width=20
        )
        self.clear_composed_button.pack(side=tk.LEFT, padx=5, pady=5)

        # Bind resize event
        self.root.bind('<Configure>', self._on_window_resize)

    def _clear_composed_level(self):
        self.added_image_indexes.clear()
        for widget in self.bottom_frame.winfo_children():
            widget.destroy()

    def _play_composed_level(self):
        if self.added_image_indexes:
            level = self.get_sample_output(self._merge_selected(self.added_image_indexes))
            level.play()

    def _merge_selected(self, indexes=None):
        if indexes is None:
            indexes = self.selected_images
        selected_scenes = [self.genomes[i].scene for i in indexes if self.genomes[i].scene]

        # Ensure all selected scenes have the same number of rows
        num_rows = len(selected_scenes[0])
        if not all(len(scene) == num_rows for scene in selected_scenes):
            raise ValueError("The selected genomes' scenes must have the same number of rows.")

        concatenated_scene = []
        for row_index in range(num_rows):
            new_row = []
            for scene in selected_scenes:
                new_row.extend(scene[row_index])
            concatenated_scene.append(new_row)

        return concatenated_scene

    def _play_all(self):
        if self.selected_images:
            level = self.get_sample_output(self._merge_selected())
            level.play()

    def _run_astar_agent_all(self):
        if self.selected_images:
            level = self.get_sample_output(self._merge_selected())
            level.run_astar()

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

    def _add_to_level(self, idx):
        self.added_image_indexes.append(idx)
        # Display thumbnail in bottom frame
        img = self.images[idx].copy()
        img.thumbnail((64, 64), Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(img)
        self.photo_images.append(photo)  # Prevent GC

        label = tk.Label(self.bottom_frame, image=photo)
        label.pack(side=tk.LEFT, padx=2)

    def get_sample_output(self, scene):
        tile_numbers = scene
        #print(self.id_to_char)
        #print(tile_numbers)
        char_grid = []
        for row in tile_numbers:
            char_row = "".join([self.id_to_char[num] for num in row])
            char_grid.append(char_row)

        #print(char_grid)
        level = SampleOutput(
            level = char_grid
        )
        return level

    def _play_genome(self, genome):
        level = self.get_sample_output(genome.scene)
        level.play()

    def _run_astar_agent(self, genome):
        level = self.get_sample_output(genome.scene)
        level.run_astar()

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
            self.callback_fn(selected)

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
            full_desc = self.genomes[idx].__str__()
            image_meta = self.genomes[idx].metadata()

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
