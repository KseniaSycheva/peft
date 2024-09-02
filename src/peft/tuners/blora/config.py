# Copyright 2023-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass, field
from typing import Optional

from peft.tuners.lora import LoraConfig
from peft.utils import PeftType


@dataclass
class BLoraConfig(LoraConfig):
    """
    This is the configuration class to store the configuration of a [`~peft.AdaLora`].

    Args:
        quantize (`bool`): Flag indicating whether quantization is applied.
        quantization_lmbd (`float`): Coefficient for gate loss.
        learn_gates (`bool`): Flag indicating whether to learn quantization gates.
        learn_scales (`bool`): The step of final fine-tuning.
        gamma_4_init (`float`): Initial gate 4 value.
        gamma_8_init (`float`): Initial gate 8 value.
        gamma_16_init (`float`): Initial gate 16 value.
        gamma_32_init (`float`): Initial gate 32 value.
        gates_lr (`float`): Learning rate for gates.
        scales_lr (`float`): Learning rate for scales.
        scales_type (`str`): Scaling type, one of `scale` and `range`.
        prune_rank (`bool`): Whether to reduce rank.
    """

    quantize: bool = field(default=True, metadata={"help": "If true, quantization is applied."})
    quantization_lmbd: float = field(default=0.001, metadata={"help": "Coefficient for gate loss."})
    n_bits: int = field(default=16, metadata={"help": "Maximum number of bits."})
    learn_gates: bool = field(default=True, metadata={"help": "If true, gates are optimized."})
    learn_scales: bool = field(default=False, metadata={"help": "If true, scales are optimized."})
    gamma_4_init: int = field(default=1, metadata={"help": "Initial gate 4 value."})
    gamma_8_init: float = field(default=6.0, metadata={"help": "Initial gate 8 value."})
    gamma_16_init: float = field(default=6.0, metadata={"help": "Initial gate 16 value."})
    gamma_32_init: float = field(default=6.0, metadata={"help": "Initial gate 32 value."})
    gates_lr: float = field(default=1e-2, metadata={"help": "Learning rate for gates."})
    scales_lr: float = field(default=1e-2, metadata={"help": "Learning rate for scales."})
    scales_type: str = field(default="scale", metadata={"help": "The total training steps."})
    prune_rank: bool = field(default=True, metadata={"help": "If true, rank is optimized."})

    def __post_init__(self):
        self.peft_type = PeftType.BLORA

        if self.use_dora:
            raise ValueError(f"{self.peft_type} does not support DoRA.")

        if self.loftq_config:
            raise ValueError(f"{self.peft_type} does not support LOFTQ.")

        self.target_modules = (
            set(self.target_modules) if isinstance(self.target_modules, list) else self.target_modules
        )
        # if target_modules is a regex expression, then layers_to_transform should be None
        if isinstance(self.target_modules, str) and self.layers_to_transform is not None:
            raise ValueError("`layers_to_transform` cannot be used when `target_modules` is a str.")

        # if target_modules is a regex expression, then layers_pattern should be None
        if isinstance(self.target_modules, str) and self.layers_pattern is not None:
            raise ValueError("`layers_pattern` cannot be used when `target_modules` is a str.")
