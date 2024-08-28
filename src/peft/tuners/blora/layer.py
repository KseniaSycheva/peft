from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from peft.tuners.blora.BayesianBits.anonymized_compression_package.quantization.straight_through import \
    QuantizationHijacker, Quantizer
from peft.tuners.blora.BayesianBits.anonymized_compression_package.utils import to_numpy
from peft.tuners.lora import LoraLayer


class BLoraLayer(LoraLayer, QuantizationHijacker):
    def __init__(
            self,
            base_layer: nn.Module,
            adapter_name,
            n_bits: int = 16,
            **kwargs
    ):
        QuantizationHijacker.__init__(
            self,
            method="bayesian_bits",
            n_bits=n_bits,
            **kwargs
        )
        LoraLayer.__init__(self, base_layer=base_layer, **kwargs)

        self.quantized_weights()
        self.quantized_acts()

        if not isinstance(base_layer, nn.Linear):
            raise ValueError("Only linear layers are currently supported.")

        self.lora_A = nn.ParameterDict({})
        self.lora_B = nn.ParameterDict({})

        self.lora_A_quantizer = nn.ModuleDict({})
        self.lora_B_quantizer = nn.ModuleDict({})
        self.lora_A_act_quantizer = nn.ModuleDict({})
        self.lora_AB_act_quantizer = nn.ModuleDict({})
        self.out_act_quantizer = nn.ModuleDict({})
        self.activation_quantizer = nn.ModuleDict({})
        self.weight_quantizer = nn.ModuleDict({})

        self.weight_quantizers = [self.lora_A_quantizer, self.lora_B_quantizer, self.weight_quantizer]
        self.activation_quantizers = [
            self.lora_A_act_quantizer,
            self.lora_AB_act_quantizer,
            self.activation_quantizer,
            self.out_act_quantizer,
        ]

        self._active_adapter = adapter_name

    def update_layer(self, adapter_name, r, lora_alpha, lora_dropout, init_lora_weights, **kwargs):
        if r < 0:
            # note: r == 0 is allowed for AdaLora, see #1539
            raise ValueError(f"`r` should be a positive integer or 0, but the value passed is {r}")

        self.r[adapter_name] = r
        self.lora_alpha[adapter_name] = lora_alpha
        if lora_dropout > 0.0:
            lora_dropout_layer = nn.Dropout(p=lora_dropout)
        else:
            lora_dropout_layer = nn.Identity()

        self.lora_dropout[adapter_name] = lora_dropout_layer
        self.scaling[adapter_name] = lora_alpha / r

        # Actual trainable parameters
        # Right singular vectors
        self.lora_A[adapter_name] = nn.Parameter(torch.randn(r, self.in_features))
        # Left singular vectors
        self.lora_B[adapter_name] = nn.Parameter(torch.randn(self.out_features, r))

        # Quantizers for weights
        for quantizer in self.weight_quantizers:
            quantizer[adapter_name] = Quantizer(
                method="bayesian_bits",
                n_bits=self.N,
                use_running_mean=False,
                **kwargs
            )
            quantizer[adapter_name].quantizer = quantizer[adapter_name].create_quantizer()

        # Quantizers for activations
        for quantizer in self.activation_quantizers:
            quantizer[adapter_name] = Quantizer(
                method="bayesian_bits",
                n_bits=self.N,
                use_running_mean=True,
                momentum=self.act_momentum,
                **kwargs
            )
            quantizer[adapter_name].quantizer = quantizer[adapter_name].create_quantizer()

        if init_lora_weights:
            self.reset_lora_parameters(adapter_name, True)

        self._move_adapter_to_device_of_base_layer(adapter_name)
        self.set_adapter(self.active_adapters)

    def reset_lora_parameters(self, adapter_name, init_lora_weights):
        if init_lora_weights is False:
            return

        if adapter_name in self.lora_A.keys():
            nn.init.normal_(self.lora_A[adapter_name], mean=0.0, std=0.02)
            nn.init.normal_(self.lora_B[adapter_name], mean=0.0, std=0.02)

    def get_params(self, adapter_name: str):
        if not self.training and self.cached_params:
            return self.cached_params

        weight, bias = self.get_weight_bias()

        if self._quant_w:
            weight = self.weight_quantizer[adapter_name](weight)
            bias = self.weight_quantizer[adapter_name](bias)

        if not self.training and self.cached_params is None:
            self.cached_params = (
                torch.Tensor(to_numpy(weight)).to(weight.device),
                torch.Tensor(to_numpy(bias)).to(bias.device)
                if bias is not None
                else None,
            )

        return weight, bias

    def get_weight_bias(self):
        if isinstance(self.base_layer, nn.Linear):
            bias = None
            if hasattr(self.base_layer, "bias"):
                bias = self.base_layer.bias
            return self.base_layer.weight, bias
        raise ValueError("Only linear layers are currently supported.")

    def get_lora_params(self, active_adapter):
        lora_A_weight, lora_B_weight = self.lora_A[active_adapter], self.lora_B[active_adapter]

        if self._quant_w:
            weight_A = self.lora_A_quantizer[active_adapter](lora_A_weight)
            weight_B = self.lora_B_quantizer[active_adapter](lora_B_weight)
        else:
            print('no quantization')
            weight_A, weight_B = lora_A_weight, lora_B_weight

        if not self.training and self.cached_params is None:
            self.cached_params = (
                torch.Tensor(to_numpy(weight_A)).to(weight_A.device),
                torch.Tensor(to_numpy(weight_B)).to(weight_B.device),
            )

        return weight_A, weight_B

    def quantize_activations(self, activations, quantizer):
        """ Quantize a single activation tensor or all activations from a layer. I'm assuming that
        we should quantize all outputs for a layer with the same quantization scheme.
        """
        if self.activation_function is not None:
            activations = self.activation_function(activations)

        if self.activation_save_target is not None:
            self.activation_save_target[
                self.activation_save_name
            ] = activations.data.cpu().numpy()

        if self._quant_a:
            activations = quantizer(activations)

            if self.activation_save_target is not None:
                self.activation_save_target[
                    self.activation_save_name + "_Q"
                    ] = activations.data.cpu().numpy()

        return activations


class BLoraNoSVDLayer(BLoraLayer):
    def __init__(
            self,
            base_layer,
            adapter_name,
            r: int = 0,
            lora_alpha: int = 1,
            lora_dropout: float = 0.0,
            fan_in_fan_out: bool = False,
            init_lora_weights: bool = True,
            **kwargs
    ):
        super().__init__(
            base_layer,
            adapter_name,
            **kwargs
        )
        self.fan_in_fan_out = fan_in_fan_out
        self.update_layer(adapter_name, r, lora_alpha, lora_dropout, init_lora_weights, **kwargs)

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        if self.disable_adapters or self.merged:
            result = self.base_layer(x, *args, **kwargs)
        else:
            result = None
            for active_adapter in self.active_adapters:
                if active_adapter not in self.lora_A.keys() or self.r[active_adapter] <= 0:
                    continue

                weight, bias = self.get_params(active_adapter)
                weight_a, weight_b = self.get_lora_params(active_adapter)

                def T(w):
                    return w.transpose(0, 1) if self.fan_in_fan_out else w

                # TODO: add handling / exception handling for other layers
                # Right now assumes that base_layer is linear
                result = F.linear(x, T(weight), bias=bias)
                result = self.quantize_activations(result, self.activation_quantizer[active_adapter])
                lora_A_act = self.lora_dropout[active_adapter](x) @ weight_a.transpose(0, 1)
                lora_A_act = self.quantize_activations(lora_A_act, self.lora_A_act_quantizer[active_adapter])
                lora_act = lora_A_act @ weight_b.transpose(0, 1)
                lora_act = self.quantize_activations(lora_act, self.lora_AB_act_quantizer[active_adapter])
                result += lora_act * self.scaling[active_adapter]
                result = self.quantize_activations(result, self.out_act_quantizer[active_adapter])

            if result is None:
                result = self.base_layer(x, *args, **kwargs)

        return result


class BLoraSVDLayer(BLoraLayer):
    def __init__(
            self,
            base_layer,
            adapter_name,
            r: int = 0,
            lora_alpha: int = 1,
            lora_dropout: float = 0.0,
            fan_in_fan_out: bool = False,
            init_lora_weights: bool = True,
            **kwargs
    ):

        super().__init__(
            base_layer,
            adapter_name,
            **kwargs
        )

        self.lora_E = nn.ParameterDict({})

        self.lora_E_quantizer = nn.ModuleDict({})
        self.lora_E_act_quantizer = nn.ModuleDict({})

        self.weight_quantizers.append(self.lora_E_quantizer)
        self.activation_quantizers.append(self.lora_E_act_quantizer)

        self.fan_in_fan_out = fan_in_fan_out

        self.update_layer(adapter_name, r, lora_alpha, lora_dropout, init_lora_weights, **kwargs)

    def update_layer(self, adapter_name, r, lora_alpha, lora_dropout, init_lora_weights, **kwargs):
        # Singular values
        self.lora_E[adapter_name] = nn.Parameter(torch.randn(r, 1))

        super().update_layer(adapter_name, r, lora_alpha, lora_dropout, init_lora_weights, **kwargs)

    def reset_lora_parameters(self, adapter_name, init_lora_weights):
        super().reset_lora_parameters(adapter_name, init_lora_weights)
        if adapter_name in self.lora_A.keys():
            nn.init.normal_(self.lora_E[adapter_name])

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        if self.disable_adapters or self.merged:
            result = self.base_layer(x, *args, **kwargs)
        else:
            result = None
            for active_adapter in self.active_adapters:
                if active_adapter not in self.lora_A.keys() or self.r[active_adapter] <= 0:
                    continue

                weight, bias = self.get_params(active_adapter)
                weight_a, weight_b, weight_e = self.get_lora_params(active_adapter)

                def T(w):
                    return w.transpose(0, 1) if self.fan_in_fan_out else w

                # TODO: add handling / exception handling for other layers
                # Right now assumes that base_layer is linear
                result = F.linear(x, T(weight), bias=bias)
                result = self.quantize_activations(result, self.activation_quantizer[active_adapter])
                lora_A_act = self.lora_dropout[active_adapter](x) @ weight_a.transpose(0, 1)
                lora_A_act = (self.quantize_activations(lora_A_act,
                                                        self.lora_A_act_quantizer[active_adapter]) *
                              weight_e.transpose(0, 1))
                lora_A_act = self.quantize_activations(lora_A_act, self.lora_E_act_quantizer[active_adapter])
                lora_act = lora_A_act @ weight_b.transpose(0, 1)
                lora_act = self.quantize_activations(lora_act, self.lora_AB_act_quantizer[active_adapter])
                result += lora_act * self.scaling[active_adapter]
                result = self.quantize_activations(result, self.out_act_quantizer[active_adapter])

            if result is None:
                result = self.base_layer(x, *args, **kwargs)

        return result

    def get_lora_params(self, active_adapter):
        lora_A_weight, lora_B_weight, lora_E_weight = (self.lora_A[active_adapter],
                                                       self.lora_B[active_adapter], self.lora_E[active_adapter])

        if self._quant_w:
            weight_A = self.lora_A_quantizer[active_adapter](lora_A_weight)
            weight_B = self.lora_B_quantizer[active_adapter](lora_B_weight)
            weight_E = self.lora_E_quantizer[active_adapter](lora_E_weight)
        else:
            print('no quantization')
            weight_A, weight_B, weight_E = lora_A_weight, lora_B_weight, lora_E_weight

        if not self.training and self.cached_params is None:
            self.cached_params = (
                torch.Tensor(to_numpy(weight_A)).to(weight_A.device),
                torch.Tensor(to_numpy(weight_B)).to(weight_B.device),
                torch.Tensor(to_numpy(weight_E)).to(weight_E.device),
            )

        return weight_A, weight_B, weight_E
