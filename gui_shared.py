import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import json

class ParentBuilder:
    def __init__(self, master):
        self.master = master
        master.title("Caption Builder")
        
        self.all_phrases = []
        self.selected_phrases = set()
        
        # Frame for checkboxes
        self.checkbox_frame = ttk.Frame(master)
        self.checkbox_frame.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.load_button = ttk.Button(self.checkbox_frame, text="Load Data", command=self.load_data)
        self.load_button.pack(anchor=tk.E)
        
        self.checkbox_canvas = tk.Canvas(self.checkbox_frame)
        self.checkbox_scrollbar = ttk.Scrollbar(self.checkbox_frame, orient=tk.VERTICAL, command=self.checkbox_canvas.yview)
        self.checkbox_inner_frame = ttk.Frame(self.checkbox_canvas)

        self.checkbox_inner_frame.bind("<Configure>", lambda e: self.checkbox_canvas.configure(scrollregion=self.checkbox_canvas.bbox("all")))
        self.checkbox_canvas.create_window((0, 0), window=self.checkbox_inner_frame, anchor="nw")
        self.checkbox_canvas.configure(yscrollcommand=self.checkbox_scrollbar.set)

        self.checkbox_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.checkbox_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
 
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

    def create_checkboxes(self):
        for widget in self.checkbox_inner_frame.winfo_children():
            widget.destroy()
        
        self.checkbox_vars.clear()
        
        # Define the specific phrases and their order
        predefined_phrases = [
            ("Level Type", ["overworld level", "underworld level"]),
            ("Sky Type", ["blue sky", "night sky"]),
            ("Floor Type", ["no floor", "full floor", "floor with gaps"])
        ]
        
        # Add predefined phrases first with separators
        for group_name, phrases in predefined_phrases:
            group_label = ttk.Label(self.checkbox_inner_frame, text=group_name, font=("Arial", 10, "bold"))
            group_label.pack(anchor=tk.W, pady=(10, 0))
            for phrase in phrases:
                if phrase in self.all_phrases:
                    var = tk.BooleanVar(value=False)
                    checkbox = ttk.Checkbutton(self.checkbox_inner_frame, text=phrase, variable=var, command=self.update_caption)
                    checkbox.pack(anchor=tk.W)
                    self.checkbox_vars[phrase] = var
                    self.all_phrases.remove(phrase)
        
        # Group remaining phrases by common patterns
        def group_phrases_by_pattern(pattern):
            #Special case for background tree
            if pattern == "tree":
                return [phrase for phrase in self.all_phrases if pattern in phrase and "giant" not in phrase] # Exclude "giant tree platform" 
            else:
                return [phrase for phrase in self.all_phrases if pattern in phrase]
        
        patterns = ["cloud", "tree", "bush", "hill", 
                    "pipe", "coin", # "brick ledge", 
                    "cannon", # "obstacle", 
                    "girder", "question block", "solid block", "metal block", "mushroom", "giant tree", "brick block",
                    "bullet bill", "koopa", "goomba", "piranha plant", "spiny", "hammer bro", "helmet"]
        
        for pattern in patterns:
            grouped_phrases = group_phrases_by_pattern(pattern)
            if grouped_phrases:
                group_label = ttk.Label(self.checkbox_inner_frame, text=f"{pattern.capitalize()} Phrases", font=("Arial", 10, "bold"))
                group_label.pack(anchor=tk.W, pady=(10, 0))
                for phrase in grouped_phrases:
                    var = tk.BooleanVar(value=False)
                    checkbox = ttk.Checkbutton(self.checkbox_inner_frame, text=phrase, variable=var, command=self.update_caption)
                    checkbox.pack(anchor=tk.W)
                    self.checkbox_vars[phrase] = var
                    self.all_phrases.remove(phrase)
        
        # Add a separator for remaining phrases
        remaining_label = ttk.Label(self.checkbox_inner_frame, text="Other Phrases", font=("Arial", 10, "bold"))
        remaining_label.pack(anchor=tk.W, pady=(10, 0))
        
        # Add the remaining phrases
        for phrase in self.all_phrases:
            var = tk.BooleanVar(value=False)
            checkbox = ttk.Checkbutton(self.checkbox_inner_frame, text=phrase, variable=var, command=self.update_caption)
            checkbox.pack(anchor=tk.W)
            self.checkbox_vars[phrase] = var
    