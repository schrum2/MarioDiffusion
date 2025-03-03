import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import json
import torch
from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler
from PIL import Image, ImageTk
import sys
from gui_shared import ParentBuilder

class CaptionBuilder(ParentBuilder):
    def __init__(self, master):
        super().__init__(master) 
        
        # Load Stable Diffusion Model
        self.pipe = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            custom_pipeline="lpw_stable_diffusion",
            torch_dtype=torch.float16,
            safety_checker=None,
            requires_safety_checker=False
        )

        if torch.cuda.is_available(): self.pipe.to("cuda")
        
        self.pipe.scheduler = EulerDiscreteScheduler.from_config(self.pipe.scheduler.config)
        
        # Frame for caption display
        self.caption_frame = ttk.Frame(master)
        self.caption_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.caption_label = ttk.Label(self.caption_frame, text="Constructed Caption:", font=("Arial", 12, "bold"))
        self.caption_label.pack(pady=5)
        
        self.caption_text = tk.Text(self.caption_frame, height=5, wrap=tk.WORD, state=tk.DISABLED)
        self.caption_text.pack() #fill=tk.BOTH, expand=True, padx=10, pady=5)
                
        self.negative_prompt_label = ttk.Label(self.caption_frame, text="Negative Prompt:")
        self.negative_prompt_label.pack()
        
        self.negative_prompt_entry = ttk.Entry(self.caption_frame)
        self.negative_prompt_entry.pack()
        
        self.num_images_label = ttk.Label(self.caption_frame, text="Number of Images:")
        self.num_images_label.pack()        
        self.num_images_entry = ttk.Entry(self.caption_frame)
        self.num_images_entry.pack()
        self.num_images_entry.insert(0, "1")

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
        
        self.height_label = ttk.Label(self.caption_frame, text="Height:")
        self.height_label.pack()
        self.height_entry = ttk.Entry(self.caption_frame)
        self.height_entry.pack()
        self.height_entry.insert(0, "256")
        
        self.width_label = ttk.Label(self.caption_frame, text="Width:")
        self.width_label.pack()
        self.width_entry = ttk.Entry(self.caption_frame)
        self.width_entry.pack()
        self.width_entry.insert(0, "256")
        
        self.generate_button = ttk.Button(self.caption_frame, text="Generate Image", command=self.generate_image)
        self.generate_button.pack(pady=5)
                
        self.lora_button = ttk.Button(self.checkbox_frame, text="Load LoRA", command=self.load_lora)
        self.lora_button.pack(anchor=tk.E)

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

        self.loaded_lora_label = ttk.Label(self.caption_frame, text=f"Using LoRA: Not loaded yet")
        self.loaded_lora_label.pack()
        
    def load_lora(self, lora_model = None):
        if lora_model == None:
            lora_model = filedialog.askopenfilename(title="Select LoRA File", filetypes=[("SafeTensors", "*.safetensors")])
        if lora_model:
            # Unload any previously loaded LoRA adapters
            self.pipe.unload_lora_weights()

            self.pipe.load_lora_weights(
                pretrained_model_name_or_path_or_dict=lora_model,
                adapter_name="my_lora",
                use_safetensors=True
            )
            self.pipe.set_adapters(["my_lora"], adapter_weights=[1.0])

            self.loaded_lora_label["text"] = f"Using LoRA: {lora_model}"
    
    def update_caption(self):
        self.selected_phrases = [phrase for phrase, var in self.checkbox_vars.items() if var.get()]
        new_caption = ". ".join(self.selected_phrases) + "." if self.selected_phrases else ""
        
        self.caption_text.config(state=tk.NORMAL)
        self.caption_text.delete(1.0, tk.END)
        self.caption_text.insert(tk.END, new_caption)
        self.caption_text.config(state=tk.DISABLED)
    
    def generate_image(self):
        prompt = self.caption_text.get("1.0", tk.END).strip()
        negative_prompt = self.negative_prompt_entry.get().strip()
        num_images = int(self.num_images_entry.get())
        param_values = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "num_inference_steps": int(self.num_steps_entry.get()),
            "guidance_scale": float(self.guidance_entry.get()),
            "height": int(self.height_entry.get()),
            "width": int(self.width_entry.get()),
        }
        generator = torch.manual_seed(int(self.seed_entry.get()))
        
        self.image_inner_frame
        for widget in self.image_inner_frame.winfo_children():
            widget.destroy()

        for _ in range(num_images):
            image = self.pipe(generator=generator, **param_values).images[0]
            img_tk = ImageTk.PhotoImage(image)
            label = ttk.Label(self.image_inner_frame, image=img_tk)
            label.image = img_tk
            label.pack()

        print("Generation done")
        
root = tk.Tk()
app = CaptionBuilder(root)

if len(sys.argv) > 1:
    app.load_data(sys.argv[1])

if len(sys.argv) > 2:
    app.load_lora(sys.argv[2])

root.mainloop()
