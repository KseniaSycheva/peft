import warnings

import torch

from peft.tuners.blora.BayesianBits.anonymized_compression_package.quantization.straight_through import \
    BayesianBitsQuantizer
from peft.tuners.blora.config import BLoraConfig
from peft.tuners.blora.layer import BLoraSVDLayer, BLoraLayer
from peft.tuners.lora import LoraModel
from peft.tuners.tuners_utils import BaseTunerLayer
from peft.utils import _freeze_adapter


class BLoraModel(LoraModel):
    def __init__(self, model, config, adapter_name):
        super().__init__(model, config, adapter_name)

        # TODO: check that it is actually needed
        traininable_mode_counter = 0
        for config in self.peft_config.values():
            if not config.inference_mode:
                traininable_mode_counter += 1

        if traininable_mode_counter > 1:
            raise ValueError(
                "BLoraModel supports only 1 trainable adapter. "
                "When using multiple adapters, set inference_mode to True for all adapters except "
                "the one you want to train."
            )

        if self.peft_config[adapter_name].inference_mode:
            _freeze_adapter(self.model, adapter_name)
        else:
            self.trainable_adapter_name = adapter_name

    def __getattr__(self, name: str):
        """Forward missing attributes to the wrapped module."""
        try:
            return super().__getattr__(name)  # defer to nn.Module's logic
        except AttributeError:
            if name == "model":  # see #1892: prevent infinite recursion if class is not initialized
                raise
            return getattr(self.model, name)

    def _create_and_replace(
        self,
        blora_config: BLoraConfig ,
        adapter_name,
        target,
        target_name,
        parent,
        current_key,
    ):
        kwargs = {
            "r": blora_config.r,
            "lora_alpha": blora_config.lora_alpha,
            "lora_dropout": blora_config.lora_dropout,
            "fan_in_fan_out": blora_config.fan_in_fan_out,
            "init_lora_weights": blora_config.init_lora_weights,
        }
        # TODO: check if actually not supported
        if kwargs["loaded_in_8bit"] or kwargs["loaded_in_4bit"]:
            raise ValueError(
                "BLoraModel cannot be combined with 4- or 8-bit loading."
            )

        # If it is not an BLoraLayer, create a new module, else update it with new adapters
        if not isinstance(target, BLoraLayer):
            new_module = self._create_new_module(blora_config, adapter_name, target, **kwargs)
            if adapter_name not in self.active_adapters:
                # adding an additional adapter: it is not automatically trainable
                new_module.requires_grad_(False)
            self._replace_module(parent, target_name, new_module, target)
        else:
            target.update_layer(
                adapter_name,
                blora_config.r,
                blora_config.lora_alpha,
                blora_config.lora_dropout,
                blora_config.init_lora_weights,
            )

    @staticmethod
    def _create_new_module(blora_config: BLoraConfig, adapter_name, target, **kwargs):
        if kwargs["loaded_in_8bit"] or kwargs["loaded_in_4bit"]:
            raise ValueError(
                "BLoraModel cannot be combined with 4- or 8-bit loading."
            )

        if isinstance(target, BaseTunerLayer):
            target_base_layer = target.get_base_layer()
        else:
            target_base_layer = target

        if isinstance(target_base_layer, torch.nn.Linear):
            if kwargs["fan_in_fan_out"]:
                warnings.warn(
                    "fan_in_fan_out is set to True but the target module is `torch.nn.Linear`. "
                    "Setting fan_in_fan_out to False."
                )
                kwargs["fan_in_fan_out"] = blora_config.fan_in_fan_out = False
        else:
            raise ValueError(
                f"Target module {target} is not supported. "
                f"Currently, only `torch.nn.Linear` is supported."
            )

        if blora_config.prune_rank:
            new_module = BLoraSVDLayer(target, adapter_name, **kwargs)
        else:
            new_module = BLoraLayer(target, adapter_name, **kwargs)
        return new_module

    @staticmethod
    def _prepare_adapter_config(peft_config, model_config):
        pass

    def gate_loss(self):
        reg_term = 0.0
        for name, module in self.model.named_modules():
            if isinstance(module, BayesianBitsQuantizer):
                reg_term += module.regularizer()
        return reg_term

    def forward(self, *args, **kwargs):
        outputs = self.model.forward(*args, **kwargs)

        if (getattr(outputs, "loss", None) is not None) and isinstance(outputs.loss, torch.Tensor):
            # Calculate gates regularization
            gate_loss_weight = self.peft_config[self.trainable_adapter_name].quantization_lmbd

            if gate_loss_weight <= 0:
                raise ValueError("quantization_lmbd should be greater than 0. ")

            gate_loss = self.gate_loss()
            outputs.loss += gate_loss_weight * gate_loss
        return outputs

    def add_weighted_adapter(self, *args, **kwargs):
        """This method is not supported for AdaLoRA, use LoRA instead."""
        raise TypeError(f"{self.__class__.__name__} does not support add_weighted_adapter method.")
