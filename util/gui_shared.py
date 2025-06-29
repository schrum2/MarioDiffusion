import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import json

GUI_FONT_SIZE = 12 

class ParentBuilder:
    def __init__(self, master):
        self.master = master
        master.title("Caption Builder")
        
        self.all_phrases = []
        self.selected_phrases = set()
        
        # Frame for checkboxes
        self.checkbox_frame = ttk.Frame(master)
        self.checkbox_frame.pack(side=tk.RIGHT, fill=tk.Y)
        


        self.checkbox_canvas = tk.Canvas(self.checkbox_frame)
        self.checkbox_scrollbar = ttk.Scrollbar(self.checkbox_frame, orient=tk.VERTICAL, command=self.checkbox_canvas.yview)
        self.checkbox_inner_frame = ttk.Frame(self.checkbox_canvas)

        self.checkbox_inner_frame.bind("<Configure>", lambda e: self.checkbox_canvas.configure(scrollregion=self.checkbox_canvas.bbox("all")))
        self.checkbox_canvas.create_window((0, 0), window=self.checkbox_inner_frame, anchor="nw")
        self.checkbox_canvas.configure(yscrollcommand=self.checkbox_scrollbar.set)

        self.checkbox_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.checkbox_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
 
        self.load_button = ttk.Button(self.checkbox_frame, text="Load Data", command=self.load_data)
        self.load_button.pack(anchor=tk.E)

        self.expand_all_button = ttk.Button(self.checkbox_frame, text="Expand All", command=self.expand_all)
        self.expand_all_button.pack(anchor=tk.E)

        self.collapse_all_button = ttk.Button(self.checkbox_frame, text="Collapse All", command=self.collapse_all)
        self.collapse_all_button.pack(anchor=tk.E)

        self.checkbox_vars = {}
    
    def load_data(self, filepath = None):
        if filepath == None:
            filepath = filedialog.askopenfilename(title="Select JSON File", filetypes=[("JSON Lines", "*.jsonl")])
        if filepath:
            try:
                phrases_set = set()
                with open(filepath, 'r') as f:
                    for line in f:
                        try:
                            item = json.loads(line)
                            if 'text' in item:
                                phrases = item['text'].split('.')
                                phrases_set.update(phrase.strip() for phrase in phrases if phrase.strip())
                        except json.JSONDecodeError as e:
                            print(f"Error decoding JSON on line: {line.strip()}. Error: {e}")
                            messagebox.showerror("Error", f"Error decoding JSON on line: {line.strip()}. Error: {e}")
                
                self.all_phrases = sorted(list(phrases_set))
                self.create_checkboxes()

                return True
            except FileNotFoundError as e:
                print(f"Error loading data: {e}")
                messagebox.showerror("Error", f"Error loading data: {e}")

        return False

    def get_predefined_phrases(self):
        # This used to mean something when shared with LoRA training, but it not used now
        predefined_phrases = [ ]
        return predefined_phrases

    def create_checkboxes(self):
        for widget in self.checkbox_inner_frame.winfo_children():
            widget.destroy()
    
        self.checkbox_vars.clear()
        self.collapsible_frames = {}  # Store references to collapsible frames
    
        # Define the specific phrases and their order
        predefined_phrases = self.get_predefined_phrases()
    
        # Create a collapsible frame class
        class CollapsibleFrame(ttk.Frame):
            def __init__(self, parent, text="", *args, **kwargs):
                ttk.Frame.__init__(self, parent, *args, **kwargs)
                self.shown = False
            
                # Frame with header and toggle button
                self.header_frame = ttk.Frame(self)
                self.header_frame.pack(fill=tk.X, expand=True)
            
                # Toggle button (+ or -)
                self.toggle_button = ttk.Label(self.header_frame, text="▶", width=2)
                self.toggle_button.pack(side=tk.LEFT, padx=(0, 5))
                self.toggle_button.bind("<Button-1>", self.toggle)
            
                # Header label
                self.title_label = ttk.Label(self.header_frame, text=text, font=("Arial", GUI_FONT_SIZE, "bold"))
                self.title_label.pack(side=tk.LEFT, fill=tk.X)
                self.title_label.bind("<Button-1>", self.toggle)
            
                # Container for content
                self.content_frame = ttk.Frame(self)
            
            def toggle(self, event=None):
                if self.shown:
                    self.content_frame.pack_forget()
                    self.toggle_button.configure(text="▶")
                else:
                    self.content_frame.pack(fill=tk.X, expand=True, padx=(15, 0))
                    self.toggle_button.configure(text="▼")
                self.shown = not self.shown
                return "break"
            
            def add_item(self, phrase, var, command):
                checkbox = tk.Checkbutton(
                    self.content_frame, 
                    text=phrase, 
                    variable=var, 
                    command=command,     
                    font=("Arial", GUI_FONT_SIZE),
                    wraplength=300,  # Adjust wrap length in pixels
                    anchor="w",
                    justify="left"
                )
                checkbox.pack(anchor=tk.W, fill=tk.X)
    
        # Function to process a group of phrases
        def create_group(name, phrases_list):
            frame = CollapsibleFrame(self.checkbox_inner_frame, text=name)
            frame.pack(fill=tk.X, anchor=tk.W, pady=(5, 0))
            self.collapsible_frames[name] = frame
        
            for phrase in phrases_list:
                if phrase in self.all_phrases:
                    var = tk.BooleanVar(value=False)
                    frame.add_item(phrase, var, self.update_caption)
                    self.checkbox_vars[phrase] = var
                    self.all_phrases.remove(phrase)
        
            return frame
    
        # Add predefined phrases first
        for group_name, phrases in predefined_phrases:
            create_group(group_name, phrases)
    
        # Group remaining phrases by common patterns
        def group_phrases_by_pattern(pattern):
            # Special case for background tree
            if pattern == "tree":
                return [phrase for phrase in self.all_phrases if pattern in phrase and "giant" not in phrase]  # Exclude "giant tree platform" 
            else:
                return [phrase for phrase in self.all_phrases if pattern in phrase]
    
        patterns = self.get_patterns()
    
        for pattern in patterns:
            grouped_phrases = group_phrases_by_pattern(pattern)
            if pattern == "enem":
                pattern = "enemy"
            if grouped_phrases:
                create_group(f"{pattern.capitalize()} Phrases", grouped_phrases)
    
        # Add remaining phrases
        if self.all_phrases:
            create_group("Other Phrases", self.all_phrases.copy())

    def get_patterns(self):
        # has to be overridden in the child class
        patterns = [ ]
        return patterns
    
    def expand_all(self):
        """Expand all collapsible topic dropdowns (CollapsibleFrames) in the checkbox area."""
        if hasattr(self, 'collapsible_frames'):
            for frame in self.collapsible_frames.values():
                if not frame.shown:
                    frame.toggle()

    def collapse_all(self):
        """Collapse all collapsible topic dropdowns (CollapsibleFrames) in the checkbox area."""
        if hasattr(self, 'collapsible_frames'):
            for frame in self.collapsible_frames.values():
                if frame.shown:
                    frame.toggle()