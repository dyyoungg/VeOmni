from veomni.arguments import parse_args
from veomni.trainer.llava_trainer import VeOmniVLMArguments, VLMTrainer
import warnings
import transformers

transformers.logging.set_verbosity_error()
warnings.filterwarnings("ignore", message=".*torch.utils.checkpoint.*")

if __name__ == "__main__":
    args = parse_args(VeOmniVLMArguments)
    trainer = VLMTrainer(args)
    trainer.train()
