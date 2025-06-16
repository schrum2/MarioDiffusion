import torch
from mario_gpt import MarioLM, SampleOutput
import argparse
import os



def parse_args():
    parser = argparse.ArgumentParser(description="Generate MarioGPT levels")

    parser.add_argument("--num_collumns", type=int, default=128, help="The number of vertical collumns to generate")
    parser.add_argument("--prompt", type=str, default=None, help="A specific prompt to generate if wanted")
    parser.add_argument("--temperature", type=float, default=2.0, help="The temperature (chaos scale) input into the model")

    parser.add_argument("--batch_size", type=int, default=5, help="The number of prompts put into the model at once")
    parser.add_argument("--output_dir", type=str, default="SMB1-gpt-levels", help="The number of vertical collumns to generate")


    return parser.parse_args()


def main():
    
    args = parse_args()



    if os.path.exists(args.output_dir):
        print("Exiting. Please remove the directory or choose a different output directory.")
        exit()
    else:
        os.makedirs(args.output_dir)

    mario_lm = MarioLM()
    # use cuda to speed stuff up
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    mario_lm = mario_lm.to(device)
    

    #Prompt loading
    all_prompts=[]

    if args.prompt is None:
        pipe_prompt_options=["no pipes", "little pipes", "some pipes", "many pipes"]
        enemy_prompt_options=["no enemies", "little enemies", "some enemies", "many enemies"]
        block_prompt_options=["little blocks", "some blocks", "many blocks"]
        elevation_prompt_options=["low elevation", "high elevation"]

        for pipe_prompt in pipe_prompt_options:
            for enemy_prompt in enemy_prompt_options:
                for block_prompt in block_prompt_options:
                    for elevation_prompt in elevation_prompt_options:
                        prompt = f"{pipe_prompt}, {enemy_prompt}, {block_prompt}, {elevation_prompt}"
                        all_prompts.append(prompt) #Creates a list of all possible prompt combinations
    else:
        all_prompts.append(args.prompt)   


    #Create directories for saving
    img_directory = os.path.join(args.output_dir, "images")
    lvl_directory = os.path.join(args.output_dir, "levels")

    os.makedirs(img_directory)
    os.makedirs(lvl_directory)



    for i in range(len(all_prompts)//args.batch_size):
        #Get the subset of all prompts to generate
        batch=all_prompts[i*args.batch_size:min((i+1)*args.batch_size, len(all_prompts))]

        #Use the subset to generate levels
        generated_levels = mario_lm.sample(
            prompts=batch,
            num_steps=args.num_collumns*14,
            temperature=args.temperature,
            use_tqdm=True
        )

        for x in range(len(generated_levels)):
            #The name of the file should match its old prompt
            save_location_img=os.path.join(img_directory, all_prompts[i*args.batch_size+x])
            save_location_lvl=os.path.join(lvl_directory, all_prompts[i*args.batch_size+x])



            # save image
            generated_levels[x].img.save(f"{save_location_img}.png")

            # save text level to file
            generated_levels[x].save(f"{save_location_lvl}.txt")






if __name__ == "__main__":
    main()