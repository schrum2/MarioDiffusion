import tkinter as tk
from tkinter import ttk, filedialog
import json
import os
from PIL import Image, ImageTk
import sys

# From Gemini

class ImageBrowser:
    def __init__(self, master, filepath = None):
        self.master = master
        master.title("Image Browser")

        self.data = {}
        self.image_paths = {}  # Store image paths
        self.all_phrases = set()
        self.filtered_data = {}

        # Frame for image display
        self.image_frame = ttk.Frame(master)
        self.image_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(self.image_frame)
        self.scrollbar_y = ttk.Scrollbar(self.image_frame, orient=tk.VERTICAL, command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.scrollbar_y.set)

        self.scrollbar_y.pack(side=tk.RIGHT, fill=tk.Y)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.canvas.bind('<Configure>', self.configure_canvas)  # Handle resizing
        self.canvas.bind_all("<MouseWheel>", self._on_mouse_wheel) # Scroll with mouse wheel

        # Frame for checkboxes
        self.checkbox_frame = ttk.Frame(master)
        self.checkbox_frame.pack(side=tk.RIGHT, fill=tk.Y)

        self.load_button = ttk.Button(master, text="Load Data", command=self.load_data)
        self.load_button.pack()

        if filepath: # Loaded from command line
            self.load_data(filepath)

    def load_data(self, filepath = None):
        if filepath == None:
            filepath = filedialog.askopenfilename(title="Select JSON File", filetypes=[("JSON Lines", "*.jsonl")])
            
        if filepath:
            try:
                self.data = []
                with open(filepath, 'r') as f:
                    for line in f:
                        try:
                            item = json.loads(line)
                            self.data.append(item)
                        except json.JSONDecodeError as e:
                            print(f"Error decoding JSON on line: {line.strip()}. Error: {e}")
                            tk.messagebox.showerror("Error", f"Error decoding JSON on line: {line.strip()}. Error: {e}")

                self.image_paths = {}  # Clear any previous image paths

                # Determine the common directory
                json_dir = os.path.dirname(filepath)
                possible_image_dirs = set()
                for item in self.data:
                    image_path = os.path.join(json_dir, item['file_name'])
                    image_dir = os.path.dirname(image_path)
                    possible_image_dirs.add(image_dir)

                if len(possible_image_dirs) == 1: # if all images are in the same directory
                    self.image_directory = list(possible_image_dirs)[0]
                else: # if not, ask the user
                    self.image_directory =  filedialog.askdirectory(title="Select Directory Containing Images", initialdir=json_dir)
                    if self.image_directory == "": # if user cancels
                        return # don't continue

                # Preload images and extract all phrases
                for item in self.data:
                    image_path = os.path.join(self.image_directory, item['file_name'])  # Use the common directory
                    if os.path.exists(image_path):
                        self.image_paths[item['file_name']] = image_path
                        phrases = item['text'].split('.')
                        for phrase in phrases:
                            self.all_phrases.add(phrase.strip())

                self.all_phrases = sorted(list(self.all_phrases))
                self.filtered_data = self.data.copy()
                self.create_checkboxes()
                #self.display_images()

            except FileNotFoundError as e:
                print(f"Error loading data: {e}")
                tk.messagebox.showerror("Error", f"Error loading data: {e}")


    def create_checkboxes(self):
        # Create a canvas and a frame for the checkboxes
        self.checkbox_canvas = tk.Canvas(self.checkbox_frame)
        self.checkbox_scrollbar = ttk.Scrollbar(self.checkbox_frame, orient=tk.VERTICAL, command=self.checkbox_canvas.yview)
        self.checkbox_inner_frame = ttk.Frame(self.checkbox_canvas)

        # Configure canvas
        self.checkbox_inner_frame.bind("<Configure>", lambda e: self.checkbox_canvas.configure(scrollregion=self.checkbox_canvas.bbox("all")))
        self.checkbox_canvas.create_window((0, 0), window=self.checkbox_inner_frame, anchor="nw")
        self.checkbox_canvas.configure(yscrollcommand=self.checkbox_scrollbar.set)

        # Pack the scrollbar and canvas
        self.checkbox_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.checkbox_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Create checkboxes
        self.checkbox_vars = {}
        for phrase in self.all_phrases:
            var = tk.BooleanVar(value=False)  # Start with all boxes unchecked
            checkbox = ttk.Checkbutton(self.checkbox_inner_frame, text=phrase, variable=var, command=self.filter_images)
            checkbox.pack(anchor=tk.W)  # Align checkboxes to the left
            self.checkbox_vars[phrase] = var


    def filter_images(self):
        checked_phrases = {phrase for phrase, var in self.checkbox_vars.items() if var.get()}  # Get checked phrases
        
        if not checked_phrases:  # If no checkboxes are checked, show nothing
            self.filtered_data = []
        else:
            self.filtered_data = [
                item for item in self.data if any(phrase.strip() in checked_phrases for phrase in item['text'].split('.'))
            ]

        self.display_images()

    def display_images(self):
        self.canvas.delete("image_item")  # Clear previous images
        self.canvas.yview_moveto(0)  # Reset scrollbar to top
        y_position = 10
        self.canvas.images = []  # Keep references to prevent garbage collection

        canvas_width = self.canvas.winfo_width()  # Get the current width of the canvas

        for item in self.filtered_data:
            image_path = self.image_paths.get(item['file_name'])
            if image_path:  # Check if image exists
                try:
                    img = Image.open(image_path)
                    img.thumbnail((300, 300))  # Resize for display
                    photo = ImageTk.PhotoImage(img)

                    # Display image
                    self.canvas.create_image(10, y_position, anchor=tk.NW, image=photo, tags=("image_item",))
                    self.canvas.images.append(photo)  # Store reference

                    y_position += img.height + 10  # Move down after the image

                    # Display file name
                    self.canvas.create_text(10, y_position, anchor=tk.NW, text=item['file_name'],
                                            tags=("image_item",), font=("Arial", 10, "bold"), width=canvas_width - 20, justify=tk.LEFT)

                    y_position += 20  # Move down after the file name

                    # Display caption
                    self.canvas.create_text(10, y_position, anchor=tk.NW, text=item['text'],
                                            tags=("image_item",), width=canvas_width - 20, justify=tk.LEFT)

                    y_position += 40  # Spacing between images
                except Exception as e:
                    print(f"Error displaying image {image_path}: {e}")

        self.canvas.config(scrollregion=self.canvas.bbox("all"))

    def configure_canvas(self, event):
        self.canvas.config(scrollregion=self.canvas.bbox("all"))

    def _on_mouse_wheel(self, event):
        self.canvas.yview_scroll(-1*(event.delta//120), "units")

root = tk.Tk()
filepath = None
if len(sys.argv) > 1:
    filepath = sys.argv[1]
app = ImageBrowser(root, filepath)
root.mainloop()
