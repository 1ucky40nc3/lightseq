from dataclasses import dataclass, field

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