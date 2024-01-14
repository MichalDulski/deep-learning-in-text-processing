from pathlib import Path
from typing import Any

from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR


class SaveOnEndEpochTrainerCallback(TrainerCallback):
    def on_epoch_end(
        self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs: Any
    ) -> None:
        training_steps = state.global_step

        # Do not save if was not trained
        if training_steps <= 0:
            return

        save_path = Path(args.output_dir) / f"{PREFIX_CHECKPOINT_DIR}-{training_steps}"
        # Skip if checkpoint exists - no need to save
        if save_path.exists():
            return

        control.should_log = True
        control.should_evaluate = True
        control.should_save = True