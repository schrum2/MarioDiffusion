import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import json

class ParentBuilder:
    def __init__(self, master):
        self.master = master
        master.title("Caption Builder")
        
        self.all_phrases = []
        self.to_delete = {}
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
        
        # Replace the original phrase grouping with dropdown-based builders
        self.create_phrase_builders()
        
        # Add a separator for remaining phrases
        remaining_label = ttk.Label(self.checkbox_inner_frame, text="Other Phrases", font=("Arial", 10, "bold"))
        remaining_label.pack(anchor=tk.W, pady=(10, 0))
        
        # Add the remaining phrases
        for phrase in self.all_phrases:
            var = tk.BooleanVar(value=False)
            checkbox = ttk.Checkbutton(self.checkbox_inner_frame, text=phrase, variable=var, command=self.update_caption)
            checkbox.pack(anchor=tk.W)
            self.checkbox_vars[phrase] = var

    def create_phrase_builders(self):
        """Create dropdown-based phrase builders for common patterns"""
        # List of entity types to create builders for
        patterns = ["cloud", "tree", "bush", "pipe", "coin", "cannon", 
                  "girder", "question block", "solid block", "metal block", "mushroom", 
                  "giant tree", "brick block", "bullet bill", "koopa", "goomba", 
                  "piranha plant", "spiny", "hammer bro", "helmet"]
        
        # Filter out patterns that don't have any phrases
        filtered_patterns = []
        for pattern in patterns:
            # Special case for background tree
            if pattern == "tree":
                phrases = [phrase for phrase in self.all_phrases if pattern in phrase and "giant" not in phrase]
            else:
                phrases = [phrase for phrase in self.all_phrases if pattern in phrase]
                
            if phrases:
                filtered_patterns.append((pattern, phrases))
        
        # Create a builder for each entity type
        for pattern, phrases in filtered_patterns:
            self.create_phrase_builder_section(pattern, phrases)

    def create_phrase_builder_section(self, entity_type, phrases):
        """Create a phrase builder section for a specific entity type"""
        # Create a frame for this entity type
        entity_frame = ttk.LabelFrame(self.checkbox_inner_frame, text=f"{entity_type.capitalize()} Phrases")
        entity_frame.pack(fill=tk.X, pady=(10, 5), padx=5, anchor=tk.W)
        
        # Analyze phrases to extract available options
        quantities = set()
        arrangements = set()
        locations = set()
        
        for phrase in phrases:
            parts = phrase.split(" in the ")
            main_part = parts[0].strip()
            
            # Extract quantity
            if main_part.startswith("a few"):
                quantities.add("a few")
                main_part = main_part[len("a few"):].strip()
            elif main_part.startswith("several"):
                quantities.add("several")
                main_part = main_part[len("several"):].strip()
            elif main_part.startswith("a "):
                quantities.add("a")
                main_part = main_part[len("a"):].strip()
            
            # Extract arrangement
            if " clustered" in main_part:
                arrangements.add("clustered")
                main_part = main_part.replace(" clustered", "")
            elif " scattered" in main_part:
                arrangements.add("scattered")
                main_part = main_part.replace(" scattered", "")
            elif " in a horizontal line" in main_part:
                arrangements.add("in a horizontal line")
                main_part = main_part.replace(" in a horizontal line", "")
            elif " in a vertical line" in main_part:
                arrangements.add("in a vertical line")
                main_part = main_part.replace(" in a vertical line", "")
            
            # Extract locations
            if len(parts) > 1:
                location_parts = parts[1].split(" and ")
                for loc in location_parts:
                    locations.add(loc.strip())
        
        # Add empty options
        quantities = [""] + sorted(list(quantities))
        arrangements = [""] + sorted(list(arrangements))
        locations = [""] + sorted(list(locations))
        
        # Create variables to store selections
        quantity_var = tk.StringVar(value="")
        arrangement_var = tk.StringVar(value="")
        location_vars = [tk.StringVar(value="") for _ in range(min(3, len(locations)))]
        
        # Create the layout with labels and dropdowns
        controls_frame = ttk.Frame(entity_frame)
        controls_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Quantity selection
        ttk.Label(controls_frame, text="Quantity:").grid(row=0, column=0, sticky=tk.W, padx=(0, 5))
        quantity_dropdown = ttk.Combobox(controls_frame, textvariable=quantity_var, values=quantities, width=10)
        quantity_dropdown.grid(row=0, column=1, sticky=tk.W, padx=5)
        
        # Arrangement selection
        ttk.Label(controls_frame, text="Arrangement:").grid(row=0, column=2, sticky=tk.W, padx=5)
        arrangement_dropdown = ttk.Combobox(controls_frame, textvariable=arrangement_var, values=arrangements, width=15)
        arrangement_dropdown.grid(row=0, column=3, sticky=tk.W, padx=5)
        
        # Location selection
        ttk.Label(controls_frame, text="Locations:").grid(row=1, column=0, sticky=tk.W, padx=(0, 5), pady=(5, 0))
        
        # Create location dropdowns
        location_frame = ttk.Frame(controls_frame)
        location_frame.grid(row=1, column=1, columnspan=3, sticky=tk.W, pady=(5, 0))
        
        for i, var in enumerate(location_vars):
            location_dropdown = ttk.Combobox(location_frame, textvariable=var, values=locations, width=12)
            location_dropdown.pack(side=tk.LEFT, padx=(0 if i == 0 else 5, 0))
        
        # Preview section
        preview_frame = ttk.Frame(entity_frame)
        preview_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(preview_frame, text="Preview:").pack(side=tk.LEFT, padx=(0, 5))
        preview_label = ttk.Label(preview_frame, text="", font=("Arial", 9, "italic"))
        preview_label.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # Function to update the preview text as selections change
        def update_preview(*args):
            # Out with the old before in with the new
            if entity_type in self.to_delete:
                var = tk.BooleanVar(value=False) 
                self.checkbox_vars[self.to_delete[entity_type]] = var

            # Get current selections
            quantity = quantity_var.get()
            arrangement = arrangement_var.get()
            selected_locations = [var.get() for var in location_vars if var.get()]
            
            # Determine entity name with correct pluralization
            entity_name = entity_type
            if quantity in ["a few", "several"]:
                if entity_name == "bush":
                    entity_name = "bushes"
                elif entity_name.endswith("y") and entity_name not in ["spiny"]:
                    entity_name = entity_name[:-1] + "ies"
                elif entity_name == "tree":
                    entity_name = "trees"
                else:
                    entity_name = entity_name + "s"
            
            # Build the phrase
            phrase = ""
            if quantity:
                phrase += quantity + " " + entity_name
                
                if arrangement:
                    phrase += " " + arrangement
                    
                if selected_locations:
                    phrase += " in the " + " and ".join(selected_locations)
            
            preview_label.config(text=phrase)
        
        # Add to selection button
        def add_to_selection():
            phrase = preview_label.cget("text")
            if phrase:
                # Create a new checkbox for this phrase
                var = tk.BooleanVar(value=True)  # Default to checked
                self.checkbox_vars[phrase] = var
                self.to_delete[entity_type] = phrase
                
                # Update the caption (which will trigger refresh in child class)
                self.update_caption()
                
                # Show confirmation
                preview_label.config(text=f"Added: {phrase}")
                
                # Reset dropdowns after short delay
                #self.master.after(1000, lambda: [
                #    quantity_var.set(""),
                #    arrangement_var.set(""),
                #    [var.set("") for var in location_vars]
                #])
        
        add_button = ttk.Button(entity_frame, text="Add to Selection", command=add_to_selection)
        add_button.pack(anchor=tk.E, padx=5, pady=(0, 5))
        
        # Bind update function to all variables
        quantity_var.trace_add("write", update_preview)
        arrangement_var.trace_add("write", update_preview)
        for var in location_vars:
            var.trace_add("write", update_preview)
        
        # Remove phrases in this group from all_phrases to avoid duplication
        for phrase in phrases:
            if phrase in self.all_phrases:
                self.all_phrases.remove(phrase)