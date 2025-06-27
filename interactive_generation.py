import torch
import webbrowser

class InteractiveGeneration:
    def __init__(self, input_parameters, default_parameters=None):
        self.input_parameters = input_parameters
        self.default_parameters = default_parameters or {}

    def start(self):
        while True:
            #try:
                param_values = dict()
                for param, param_type in self.input_parameters.items():
                    default = self.default_parameters.get(param, "")
                    prompt = f"{param}"
                    if default != "":
                        prompt += f" [default value: {default}]"
                    prompt += ": "
                    user_input = input(prompt)
                    if user_input == "q":
                        quit()
                    if user_input == "":
                        if param == "end_seed" and "start_seed" in param_values:
                            # Special case: end_seed defaults to start_seed
                            param_values[param] = param_values["start_seed"]
                        else:
                            param_values[param] = default
                    else:
                        param_values[param] = param_type(user_input)

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
                    elif image:
                        image.show()
            #except Exception as e:
            #    print(f"Error: {e}")
            #    continue

    def get_extra_params(self, param_values): # Default nothing
        return dict()
