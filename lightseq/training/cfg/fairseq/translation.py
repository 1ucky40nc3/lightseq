from dataclasses import dataclass, field

from fairseq.tasks import register_task
from fairseq.tasks.translation import TranslationTask
from fairseq.tasks.translation import TranslationConfig


class LSTranslationConfig(TranslationConfig):
    max_tokens: int = field(
        default=128, 
        metadata={"help": "max number of tokens in batch"}
    )
    fp16: bool = field(
        default=False, 
        metadata={"help": "state wether to train with half precision"}
    )    
    device_id: int = field(
        default=0,
        metadata={
            "help": "device id of your accelerator."
        },
    )


@register_task("ls_translation", dataclass=LSTranslationConfig)
class LSTranslationTask(TranslationTask):
    def __init__(self, cfg: LSTranslationConfig, src_dict, tgt_dict):
        super().__init__(cfg)