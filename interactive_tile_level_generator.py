import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
import json
import torch
import gc
from PIL import ImageTk
import sys
from util.gui_shared import ParentBuilder
from models.text_diffusion_pipeline import TextConditionalDDPMPipeline
from level_dataset import visualize_samples, convert_to_level_format
from util.sampler import SampleOutput
from captions.caption_match import compare_captions
from create_ascii_captions import assign_caption, extract_tileset

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

        self.width_label = ttk.Label(self.caption_frame, text="Width (in tiles):")
        self.width_label.pack()
        self.width_entry = ttk.Entry(self.caption_frame)
        self.width_entry.pack()
        self.width_entry.insert(0, "16")
                
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
        
        # Bind mousewheel scrolling to the scrollbar for image_inner_frame
        self.image_canvas.bind_all("<MouseWheel>", lambda event: self.image_canvas.yview_scroll(-1 * (event.delta // 120), "units"))
        
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
                    "pipe", "coin", "platform", "tower", #"wall",
                    "cannon", "staircase", "rectangular", "irregular",
                    "question block", "enem"]
        return patterns

    def load_data(self, filepath = None):
        if filepath == None:
            filepath = filedialog.askopenfilename(title="Select JSON File", filetypes=[("JSON", "*.json")])
        if filepath:
            _, self.id_to_char, self.char_to_id, self.tile_descriptors = extract_tileset(tileset_path)

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

        param_values = {
            # "captions" : sample_caption_tokens, # Added below if prompt is not None
            "num_inference_steps": int(self.num_steps_entry.get()),
            "guidance_scale": float(self.guidance_entry.get()),
            "width": int(self.width_entry.get()),
            "output_type" : "tensor",
            "batch_size" : 1
        }

        # Include caption if desired
        if prompt.strip() != "":
            sample_captions = [prompt] # batch of size 1
            sample_caption_tokens = self.pipe.text_encoder.tokenizer.encode_batch(sample_captions)
            sample_caption_tokens = torch.tensor(sample_caption_tokens).to(self.device)

            param_values["captions"] = sample_caption_tokens

        generator = torch.Generator(self.device).manual_seed(int(self.seed_entry.get()))
        
        self.image_inner_frame
        for widget in self.image_inner_frame.winfo_children():
            widget.destroy()

        self.current_levels = []

        # Debugging print statements to trace the issue
        print("Starting image generation...")
        self.image_inner_frame.update_idletasks()  # Force an update to ensure the frame is fully rendered
        frame_width = self.image_inner_frame.winfo_width()
        print(f"Frame width after update_idletasks: {frame_width}")

        # Use a cached frame width if available and valid
        if hasattr(self, 'cached_frame_width') and self.cached_frame_width > 1:
            frame_width = self.cached_frame_width
            print(f"Using cached frame width: {frame_width}")
        elif frame_width <= 1:  # If the width is invalid or too small
            frame_width = self.image_canvas.winfo_width() // 2  # Use third of the parent canvas width as a fallback
            print(f"Frame width was invalid, using third of canvas width: {frame_width}")
        else:
            # Cache the valid frame width for future use
            self.cached_frame_width = frame_width
            print(f"Caching frame width: {frame_width}")

        for i in range(num_images):
            print(f"Generating image {i + 1} of {num_images}...")
            images = self.pipe(generator=generator, **param_values).images
            self.current_levels.append(images[0].cpu().detach().numpy())

            sample_tensor = images[0].unsqueeze(0)
            sample_indices = convert_to_level_format(sample_tensor)
            scene = sample_indices[0].tolist()
            actual_caption = assign_caption(scene, self.id_to_char, self.char_to_id, self.tile_descriptors, False, False)

            compare_score, exact_matches, partial_matches, excess_phrases = compare_captions(prompt, actual_caption, return_matches=True)

            img_frame = ttk.Frame(self.image_inner_frame)
            img_frame.pack(pady=10)

            img_tk = ImageTk.PhotoImage(visualize_samples(images))
            print(f"Image {i + 1} dimensions: width={img_tk.width()}, height={img_tk.height()}")

            # Check if the image width exceeds the frame width and scale it down if necessary
            if img_tk.width() > frame_width:
                scale_factor = frame_width / img_tk.width()
                new_width = frame_width
                new_height = int(img_tk.height() * scale_factor)
                img_tk = img_tk._PhotoImage__photo.subsample(img_tk.width() // new_width, img_tk.height() // new_height)
                print(f"Image {i + 1} scaled to: width={new_width}, height={new_height}")

            label = ttk.Label(img_frame, image=img_tk)
            label.image = img_tk
            label.pack()

            # Create a Text widget to allow colored text
            caption_text = tk.Text(img_frame, wrap=tk.WORD, width=40, height=5, state=tk.DISABLED)
            caption_text.pack(pady=(5, 10))

            # Enable editing temporarily to insert text
            caption_text.config(state=tk.NORMAL)

            # Define tags for different colors
            caption_text.tag_configure("green", foreground="green")
            caption_text.tag_configure("yellow", foreground="#CCCC00")  # Darker yellow
            caption_text.tag_configure("red", foreground="red")

            # Insert text with tags
            for phrase in exact_matches:
                caption_text.insert(tk.END, phrase + ". ", "green")
            for phrase in partial_matches:
                caption_text.insert(tk.END, phrase + ". ", "yellow")
            for phrase in excess_phrases:
                caption_text.insert(tk.END, phrase + ". ", "red")

            # Disable editing again
            caption_text.config(state=tk.DISABLED)

            # And score
            #score_label = ttk.Label(img_frame, text=f"Comparison Score: {compare_score}", wraplength=300)
            #score_label.pack(pady=(5, 10))  # Add padding: 5px top, 10px bottom

            # Check if the scene is wider than 16 tiles and process segments if necessary
            avg_segment_score = None
            if len(scene[0]) > 16:
                from captions.caption_match import process_scene_segments
                avg_segment_score, _, _ = process_scene_segments(
                    scene=scene,
                    segment_width=16,
                    prompt=prompt,
                    id_to_char=self.id_to_char,
                    char_to_id=self.char_to_id,
                    tile_descriptors=self.tile_descriptors,
                    describe_locations=False,
                    describe_absence=False
                )

            # Update the score label text
            if avg_segment_score is not None:
                score_label_text = f"""Comparison Score: {compare_score}
Average Segment Score: {avg_segment_score}"""
            else:
                score_label_text = f"Comparison Score: {compare_score}"

            score_label = ttk.Label(img_frame, text=score_label_text, wraplength=300)
            score_label.pack(pady=(5, 10))  # Add padding: 5px top, 10px bottom
    
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

            del images, sample_tensor, sample_indices, scene  # Delete unused tensors
            torch.cuda.empty_cache()  # Clear the cache
            gc.collect()  # Force garbage collection

        print("Image generation completed.")
        #print(self.current_levels)

    def get_sample_output(self, idx):
        tensor = torch.tensor(self.current_levels[idx])
        tile_numbers = torch.argmax(tensor, dim=0).numpy()
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
      
    def play_level(self, idx):
        level = self.get_sample_output(idx)
        level.play()

    def use_astar(self, idx):
        level = self.get_sample_output(idx)
        level.run_astar()
  
root = tk.Tk()
app = CaptionBuilder(root)

global tileset_path
tileset_path = '..\TheVGLC\Super Mario Bros\smb.json'
if len(sys.argv) > 3:
    tileset_path = sys.argv[3]

if len(sys.argv) > 1:
    app.load_data(sys.argv[1])

if len(sys.argv) > 2:
    app.load_model(sys.argv[2])

root.mainloop()
