import torch
import webbrowser

class InteractiveGeneration:
    def __init__(self, input_parameters):
        self.input_parameters = input_parameters

    def start(self):
        while True:
            try:
                param_values = dict()
                for param, param_type in self.input_parameters.items():
                    param_values[param] = input(f"{param} (q to quit): ")
                    if param_values[param] == "q": quit()
                    param_values[param] = param_type(param_values[param])

                start_seed = param_values["start_seed"]
                del param_values["start_seed"]
                end_seed = param_values["end_seed"]
                del param_values["end_seed"]

                extra_params = self.get_extra_params(param_values)
                for seed in range(start_seed, end_seed+1):
                    generator = torch.Generator("cuda").manual_seed(seed)
                    image = self.generate_image(param_values, generator, **extra_params)
                    if isinstance(image, list):
                        # Assume this represents an animation
                        webbrowser.open("test.gif")
                    else:
                        image.show()
            except Exception as e:
                print(f"Error: {e}")
                continue
            
    def get_extra_params(self, param_values): # Default nothing
        return dict()
