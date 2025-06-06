import torch
from mario_gpt.lm import MarioLM, BaseMarioLM
from transformers import AutoModelWithLMHead, AutoTokenizer
import os
from dataclasses import asdict, dataclass
from typing import Any, Optional, Tuple
import numpy as np
import torch
from accelerate import Accelerator
from torch.optim.optimizer import Optimizer
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizer, get_linear_schedule_with_warmup, pipeline
from torch.optim import AdamW
from level_dataset import LevelDataset
from util.sampler import scene_to_ascii
from captions.util import extract_tileset

tileset = '..\TheVGLC\Super Mario Bros\smb.json'
json_path = "datasets\\SMB1_LevelsAndCaptions-regular-train.json"
num_tiles = "13"

def main():
    # create basic gpt model
    BASE = "distilgpt2"
    lm = AutoModelWithLMHead.from_pretrained(BASE, add_cross_attention = True)

    tokenizer = AutoTokenizer.from_pretrained(BASE)

    mario_lm = MarioLM(lm=lm, tokenizer=tokenizer)

    # create dataset
    dataset = LevelDataset(
        json_path=json_path,
        tokenizer=tokenizer,
        shuffle=True,
        mode='diff_text',
        augment=True,
        num_tiles=num_tiles,
        negative_captions=False,
        block_embeddings=None
    )

    # create training config and trainer
    config = TrainingConfig(eval_iteration=100000, output_dir="Mario-GPT2-210-context-length")
    trainer = MarioGPTTrainer(mario_lm, dataset, tokenizer=tokenizer, config=config) #I'm not sure this is the correct tokenizer

    # train for 100 iterations!
    trainer.train(50000, batch_size=4)




@dataclass
class TrainingConfig:
    gradient_accumulation_steps: int = 1
    mixed_precision: str = (
        "no"  # `no` for float32, `fp16` for automatic mixed precision
    )
    output_dir: str = (
        "Mario-GPT2-700-context-length"  # the model name locally and on the HF Hub
    )
    learning_rate: float = 5e-4
    epsilon: float = 1e-9
    lr_warmup_steps: int = 1000
    batch_size: int = 4
    total_steps: int = 50000
    mask_proportion: float = 0.0
    eval_iteration: int = 1000
    save_iteration: int = 5000

    def pretty_print(self):
        print("================== Training Config ==================")
        d = asdict(self)
        for k in d:
            print(f"{k} -- {d[k]}")
        print("================== MarioLM ==================")


class MarioGPTTrainer:
    def __init__(
        self,
        mario_lm: BaseMarioLM,
        train_dataset: LevelDataset,
        tokenizer: PreTrainedTokenizer,
        config: Optional[TrainingConfig] = None,
        optimizer: Optional[Optimizer] = None,
        lr_scheduler: Optional[Any] = None,
    ):
        self.mario_lm = mario_lm
        self.train_dataset = train_dataset
        self.tokenizer=tokenizer

        self.feature_extraction = pipeline(
            "feature-extraction",
            model="facebook/bart-base",
            tokenizer="facebook/bart-base",
            framework="pt",
        )

        self.config = config

        if config is None:
            self.config = TrainingConfig()

        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

        if optimizer is None:
            self.optimizer = self.create_optimizer(self.config)
        if lr_scheduler is None:
            self.lr_scheduler = self.create_lr_scheduler(self.config, self.optimizer)

        self.accelerator = self.create_accelerator(self.config)

    def prepare(self) -> Tuple[PreTrainedModel, Optimizer, Any]:
        return self.accelerator.prepare(
            self.mario_lm.lm, self.optimizer, self.lr_scheduler
        )

    def create_optimizer(self, config: Any) -> Optimizer:
        params = self.mario_lm.lm.parameters()
        return AdamW(params, lr=config.learning_rate, eps=config.epsilon)

    def create_lr_scheduler(self, config: Any, optimizer: Optimizer) -> Any:
        return get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=config.lr_warmup_steps,
            num_training_steps=config.total_steps,
        )

    def create_accelerator(self, config: Any) -> Accelerator:
        return Accelerator(
            mixed_precision=config.mixed_precision,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            log_with="tensorboard",
            project_dir=config.output_dir,
        )

    def unwrap(self) -> BaseMarioLM:
        return MarioLM(
            lm=self.accelerator.unwrap(self.mario_lm.lm),
            tokenizer=self.mario_lm.tokenizer,
            context_len=self.mario_lm.context_len,
            prompter=self.mario_lm.prompter,
        )

    def sample_from_dataset(
        self, dataset: Dataset, batch_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        indices = list(
            torch.randint(low=0, high=len(dataset), size=(batch_size,)).long()
        )
        return dataset[indices]

    def train_iter(
        self,
        accelerator: Accelerator,
        model: PreTrainedModel,
        train_dataset: LevelDataset,
        optimizer: Any,
        scheduler: Any,
        batch_size: int = 4,
    ):
        device = accelerator.device
        total_train_loss = 0
        indices = list(
            torch.randint(low=0, high=len(train_dataset), size=(batch_size,)).long()
        )

        batch = train_dataset[indices]
        one_hot_scenes, captions=batch
        if not isinstance(captions, list):
            captions = [captions]

        input_id_list, attention_mask_list = convert_one_hot_to_gpt_data(one_hot_scenes, tokenizer=self.tokenizer)


        b_input_ids = input_id_list.view(batch_size, -1).to(device)
        b_labels = input_id_list.view(batch_size, -1).to(device)
        attention_masks = attention_mask_list.to(device)

        encoder_hidden_states = None
        encoder_hidden_states = []
        for caption in captions:
            encoder_hidden_state = self.feature_extraction(caption, return_tensors="pt")[0].mean(0).to(device).view(1, -1)
            encoder_hidden_states.append(encoder_hidden_state)
        encoder_hidden_states = torch.stack(encoder_hidden_states, dim=0).view(
            batch_size, 1, -1
        )

        with accelerator.accumulate(model):
            model.zero_grad()
            outputs = model(
                input_ids=b_input_ids.to(device),
                labels=b_labels,
                attention_mask=attention_masks,
                encoder_hidden_states=encoder_hidden_states,
                token_type_ids=None,
            )
            loss = outputs.loss

            batch_loss = loss.item()
            total_train_loss += batch_loss

            loss.backward()
            optimizer.step()
            scheduler.step()

        grad_dict = {}
        for n, W in model.named_parameters():
            if W.grad is not None:
                grad_dict["{}_grad".format(n)] = float(torch.sum(W.grad).item())

        return total_train_loss / batch_size, grad_dict

    def train(
        self,
        total_steps: Optional[int] = None,
        batch_size: Optional[int] = None,
    ):
        if total_steps is None:
            total_steps = self.config.total_steps
        if batch_size is None:
            batch_size = self.config.batch_size

        self.accelerator.init_trackers("mario-gpt")

        checkpoint_path = self.config.output_dir
        logdir = os.path.abspath(self.accelerator.logging_dir)

        print(f"Training for {total_steps} Iterations and batch_size {batch_size}")
        if getattr(self.config, "pretty_print", None) is not None:
            self.config.pretty_print()
        print(f"Follow tensorboard with: python -m tensorboard.main --logdir {logdir}")

        model, optimizer, lr_scheduler = self.prepare()

        bar = tqdm(np.arange(total_steps))
        model.train()

        for i in bar:
            loss, grad_dict = self.train_iter(
                self.accelerator,
                model,
                self.train_dataset,
                optimizer,
                lr_scheduler,
                batch_size,
            )
            logs = {"loss": loss, "last_lr": lr_scheduler.get_last_lr()[0]}
            bar.set_description(f"{logs}")
            self.accelerator.log({**logs, **grad_dict}, step=i)

            """if (i + 1) % self.config.eval_iteration == 0:
                print("Evaluating...")
                with torch.no_grad():
                    try:
                        if self.config.mask_proportion <= 0.0:
                            (
                                prompt,
                                _,
                                _,
                                _,
                            ) = self.mario_lm.prompter(sample_prompt=True)
                            out = self.mario_lm.sample(
                                prompts=[prompt],
                                num_steps=1400,
                                temperature=2.0,
                                use_tqdm=True,
                            )
                            draw = ImageDraw.Draw(out.img)
                            draw.text((0, 0), prompt, (0, 0, 0))
                            tracker = self.accelerator.get_tracker("tensorboard")
                            tracker.add_image(
                                "image", np.array(out.img), i, dataformats="HWC"
                            )
                    except Exception as e:
                        print("Failed to evaluate!", e)
                model.train()"""
            if (i + 1) % self.config.save_iteration == 0:
                self.mario_lm.lm.save_pretrained(os.path.join(checkpoint_path, f"iteration_{i}"))



def convert_one_hot_to_gpt_data(scenes: torch.tensor, tokenizer: PreTrainedTokenizer, 
                                expected_height: int = 16):
    """
        Method that takes in a list of one-hot encoded levels and outputs their
        input ids and attention masks for the GPT model
    """
    
    #Converts each one-hot encoded tensor into the single string that MarioGPT expects
    input_id_list=[]
    attention_mask_list=[]

    for scene in scenes:
        scene=torch.argmax(scene, dim=1)

        #We need our height to be consistant for this to work properly
        if len(scene!=expected_height):
            raise ValueError(f"Scene height {len(scene)} is not equal to the expected scene height {expected_height}")
        
        #Conversion to a list of horizantal strings
        _, id_to_char, _, _=extract_tileset(tileset)
        scene = scene_to_ascii(scene=scene, id_to_char=id_to_char)

        #Turns our list of strings(horizantal) into a list of lists of strings (vertial)
        scene = np.array([list(s) for s in scene])
        np.flip(scene.transpose(), -1)

        #Turns the whole thing into one long string
        scene = "".join(["".join(s) for s in scene])

        #Encodes text string 
        scene, _ = tokenizer(scene, return_tensors="pt")


        input_ids = scene["input_ids"].squeeze()
        attention_masks = scene["attention_mask"].squeeze()

        input_id_list.append(input_ids)
        attention_mask_list.append(attention_masks)
    

    return input_id_list, attention_mask_list






if __name__ == "__main__":
    main()