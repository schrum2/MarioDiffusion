import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
import json
import torch
from PIL import Image, ImageTk
from diffusers import UNet2DModel, UNet2DConditionModel, DDPMScheduler, DDPMPipeline
import sys
from gui_shared import ParentBuilder
from tokenizer import Tokenizer 
from models import TransformerModel
from text_diffusion_pipeline import TextConditionalDDPMPipeline
from level_dataset import visualize_samples
from sampler import SampleOutput

class CaptionBuilder(ParentBuilder):
    def __init__(self, master):
        super().__init__(master) 
                
        # Holds tensors of levels currently on display
        self.current_levels = []

        # Frame for caption display
        self.caption_frame = ttk.Frame(master)
        self.caption_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.caption_label = ttk.Label(self.caption_frame, text="Constructed Caption:", font=("Arial", 12, "bold"))
        self.caption_label.pack(pady=5)
        
        self.caption_text = tk.Text(self.caption_frame, height=5, wrap=tk.WORD, state=tk.DISABLED)
        self.caption_text.pack() 
                
        #self.negative_prompt_label = ttk.Label(self.caption_frame, text="Negative Prompt:")
        #self.negative_prompt_label.pack()
        
        #self.negative_prompt_entry = ttk.Entry(self.caption_frame)
        #self.negative_prompt_entry.pack()
        
        self.num_images_label = ttk.Label(self.caption_frame, text="Number of Images:")
        self.num_images_label.pack()        
        self.num_images_entry = ttk.Entry(self.caption_frame)
        self.num_images_entry.pack()
        self.num_images_entry.insert(0, "4")

        self.seed_label = ttk.Label(self.caption_frame, text="Random Seed:")
        self.seed_label.pack()        
        self.seed_entry = ttk.Entry(self.caption_frame)
        self.seed_entry.pack()
        self.seed_entry.insert(0, "1")

        self.num_steps_label = ttk.Label(self.caption_frame, text="Num Inference Steps:")
        self.num_steps_label.pack()
        self.num_steps_entry = ttk.Entry(self.caption_frame)
        self.num_steps_entry.pack()
        self.num_steps_entry.insert(0, "50")
        
        self.guidance_label = ttk.Label(self.caption_frame, text="Guidance Scale:")
        self.guidance_label.pack()
        self.guidance_entry = ttk.Entry(self.caption_frame)
        self.guidance_entry.pack()
        self.guidance_entry.insert(0, "7.5")
                
        self.generate_button = ttk.Button(self.caption_frame, text="Generate Image", command=self.generate_image)
        self.generate_button.pack(pady=5)
                
        self.model_button = ttk.Button(self.checkbox_frame, text="Load Model", command=self.load_model)
        self.model_button.pack(anchor=tk.E)

        # Frame for image display
        self.image_frame = ttk.Frame(master)
        self.image_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        self.image_canvas = tk.Canvas(self.image_frame)
        self.image_scrollbar = ttk.Scrollbar(self.image_frame, orient=tk.VERTICAL, command=self.image_canvas.yview)
        self.image_inner_frame = ttk.Frame(self.image_canvas)
        
        self.image_inner_frame.bind("<Configure>", lambda e: self.image_canvas.configure(scrollregion=self.image_canvas.bbox("all")))
        self.image_canvas.create_window((0, 0), window=self.image_inner_frame, anchor="nw")
        self.image_canvas.configure(yscrollcommand=self.image_scrollbar.set)
        
        self.image_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.image_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.checkbox_vars = {}

        self.loaded_model_label = ttk.Label(self.caption_frame, text=f"Using model: Not loaded yet")
        self.loaded_model_label.pack()

    def get_predefined_phrases(self):
        # Behaves differently for LoRA vs plain diffusion model
        # No phrases for plain diffusion model
        predefined_phrases = [ ]
        return predefined_phrases

    def get_patterns(self):
        # Different for LoRA and tile diffusion
        patterns = ["floor", "ceiling", 
                    "pipe", "coin", "platform", "tower", "wall",
                    "cannon", "staircase", "irregular",
                    "question block", "enem"]
        return patterns

    def load_data(self, filepath = None):
        if filepath == None:
            filepath = filedialog.askopenfilename(title="Select JSON File", filetypes=[("JSON", "*.json")])
        if filepath:
            try:
                phrases_set = set()
                with open(filepath, 'r') as f:
                    dataset = json.load(f)
                    for item in dataset:
                        phrases = item['caption'].split('.')
                        phrases_set.update(phrase.strip() for phrase in phrases if phrase.strip())
                
                self.all_phrases = sorted(list(phrases_set))
                self.create_checkboxes()

                return True
            except FileNotFoundError as e:
                print(f"Error loading data: {e}")
                messagebox.showerror("Error", f"Error loading data: {e}")

        return False
        
    def load_model(self, model = None):
        if model == None:
            model = filedialog.askopenfilename(title="Select Model Index", filetypes=[("JSON", "*.json")])
            if model: # removed model model_index.json
                model = os.path.dirname(model)
        if model:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.pipe = TextConditionalDDPMPipeline.from_pretrained(model).to(self.device)

            filename = os.path.splitext(os.path.basename(model))[0]
            self.loaded_model_label["text"] = f"Using model: {filename}"
    
    def update_caption(self):
        self.selected_phrases = [phrase for phrase, var in self.checkbox_vars.items() if var.get()]
        new_caption = ". ".join(self.selected_phrases) + "." if self.selected_phrases else ""
        
        self.caption_text.config(state=tk.NORMAL)
        self.caption_text.delete(1.0, tk.END)
        self.caption_text.insert(tk.END, new_caption)
        self.caption_text.config(state=tk.DISABLED)
    
    def generate_image(self):
        print("Generating")
        prompt = self.caption_text.get("1.0", tk.END).strip()
        num_images = int(self.num_images_entry.get())

        sample_captions = [prompt] # batch of size 1
        sample_caption_tokens = self.pipe.text_encoder.tokenizer.encode_batch(sample_captions)
        sample_caption_tokens = torch.tensor(sample_caption_tokens).to(self.device)

        param_values = {
            "captions" : sample_caption_tokens,
            "num_inference_steps": int(self.num_steps_entry.get()),
            "guidance_scale": float(self.guidance_entry.get()),
            "output_type" : "tensor",
            "batch_size" : 1
        }
        generator = torch.Generator(self.device).manual_seed(int(self.seed_entry.get()))
        
        self.image_inner_frame
        for widget in self.image_inner_frame.winfo_children():
            widget.destroy()

        self.current_levels = []
        for i in range(num_images):
            images = self.pipe(generator=generator, **param_values).images
            # Save each generated level
            self.current_levels.append(images[0].cpu().detach().numpy())
            
            # Create a frame for each image and its buttons
            img_frame = ttk.Frame(self.image_inner_frame)
            img_frame.pack(pady=10)
    
            # Display the image
            img_tk = ImageTk.PhotoImage(visualize_samples(images))
            label = ttk.Label(img_frame, image=img_tk)
            label.image = img_tk
            label.pack()
    
            # Create a frame for buttons
            button_frame = ttk.Frame(img_frame)
            button_frame.pack(pady=5)
    
            # Add Play button
            play_button = ttk.Button(
                button_frame, 
                text="Play", 
                command=lambda idx=i: self.play_level(idx)
            )
            play_button.pack(side=tk.LEFT, padx=5)
    
            # Add Use A* button
            astar_button = ttk.Button(
                button_frame, 
                text="Use A*", 
                command=lambda idx=i: self.use_astar(idx)
            )
            astar_button.pack(side=tk.LEFT, padx=5)

        print("Generation done")
        #print(self.current_levels)
      
    def play_level(self, idx):
        tensor = torch.tensor(self.current_levels[idx])
        tile_numbers = torch.argmax(tensor, dim=0).numpy()
        print(tile_numbers)
        tile_chars = sorted(tileset['tiles'].keys())
        id_to_char = {idx: char for idx, char in enumerate(tile_chars)}
        char_grid = []
        for row in tile_numbers:
            char_row = "".join([id_to_char[num] for num in row])
            char_grid.append(char_row)

        print(char_grid)
        level = SampleOutput(
            level = char_grid
        )
        level.play()

    def use_astar(self, idx):
        tensor = torch.tensor(self.current_levels[idx])
        tile_numbers = torch.argmax(tensor, dim=0).numpy()
        print(tile_numbers) 
  
root = tk.Tk()
app = CaptionBuilder(root)

if len(sys.argv) > 1:
    app.load_data(sys.argv[1])

if len(sys.argv) > 2:
    app.load_model(sys.argv[2])

global tileset
tileset_path = '..\TheVGLC\Super Mario Bros\smb.json'
if len(sys.argv) > 3:
    tileset_path = sys.argv[3]

with open(tileset_path, 'r') as f:
    tileset = json.load(f)

root.mainloop()
