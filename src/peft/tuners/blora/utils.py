import numpy as np

import torch
import torch.nn as nn
from transformers import TrainerCallback

import wandb


def get_proj_groups(layer: nn.Module, layer_n: int, prefix: str, model_name, hidden_act: torch.tensor = None,
                    is_lora: bool = False, prune_rank: bool = False):
    """
    Groups projection quantizers.
    LoRa: (input activations (possibly none), weight), (input activations (possibly none), A), (A activations, B).
    Not LoRa: (input activations (possibly none), weight).

    :param layer: projection layer (key, query or value).
    :param layer_n: number of transformer layer (from 0 to config.num_hidden_layers - 1).
    :param prefix: string that represents the parent class.
    :param model_name: name of the model.
    :param hidden_act: output of the previous layer (input of the current layer).
    :param is_lora: indicates if LoRa is used the layer.
    :param prune_rank: if rank is pruned.

    :return: grouped quantizers.
    """
    groups = []
    if model_name == 'deberta':
        input_template = f"DebertaV2Model.encoder.layer.{layer_n - 1}.output.dense.activation_quantizer"
    elif model_name == 'roberta':
        input_template = f"RobertaModel.encoder.layer.{layer_n - 1}.output.dense.activation_quantizer"
    else:
        raise ValueError("Logging not supported.")

    q_template = "{prefix}.{name}_quantizer"

    if hidden_act is not None:
        groups.append(((input_template, q_template.format(prefix=prefix, name="weight"), ),
                       [hidden_act, layer.weight_quantizer["default"]]))
        if is_lora:
            groups.append(((input_template, q_template.format(prefix=prefix, name="lora_A"), ),
                           [hidden_act, layer.lora_A_quantizer["default"]]))
    else:
        groups.append(((q_template.format(prefix=prefix, name="weight"),), [layer.weight_quantizer["default"]]))
        if is_lora:
            groups.append(((q_template.format(prefix=prefix, name="lora_A"),), [layer.lora_A_quantizer["default"]]))
            if prune_rank:
                groups.append(((q_template.format(prefix=prefix, name="lora_A_act"),
                                q_template.format(prefix=prefix, name="lora_E"),), [layer.lora_A_act_quantizer["default"],
                                                                                    layer.lora_E_quantizer["default"]]))

    if is_lora:
        if prune_rank:
            groups.append(((q_template.format(prefix=prefix, name="lora_E_act"),
                            q_template.format(prefix=prefix, name="lora_B"),),
                           [layer.lora_A_act_quantizer["default"], layer.lora_B_quantizer["default"]]))
        else:
            groups.append(((q_template.format(prefix=prefix, name="lora_A_act"),
                            q_template.format(prefix=prefix, name="lora_B"),),
                           [layer.lora_A_act_quantizer["default"], layer.lora_B_quantizer["default"]]))

    if is_lora:
        groups.append(((q_template.format(prefix=prefix, name="activation"),), [layer.activation_quantizer["default"]]))
        groups.append(((q_template.format(prefix=prefix, name="out_act_quantizer"),), [layer.out_act_quantizer["default"]]))

    return groups


def get_attention_groups(attn_layer: nn.Module, layer_n: int, prefix: str, model_name, hidden_act: torch.tensor = None,
                         quantize_only_trainable: bool = True, trainable: list[str] = None, prune_rank: bool = False):
    """
    Groups attention quantizers.

    :param attn_layer: attention layer.
    :param layer_n: number of transformer layer (from 0 to config.num_hidden_layers - 1).
    :param prefix: string that represents the parent class.
    :param model_name: name of the model.
    :param hidden_act: output of the previous layer (input of the current layer).
    :param quantize_only_trainable: if used, method applied only to LoRA blocks.
    :param trainable: modules to apply LoRA to.
    :param prune_rank: if rank is pruned.

    :return: grouped quantizers.
    """
    self_template = prefix + ".attention.self.{name}_quantizer"
    out_template = prefix + ".attention.output.dense.{name}_quantizer"
    self_attn = attn_layer.self

    groups = []
    if model_name == "deberta":
        suffix = "_proj"
    elif model_name == "roberta":
        suffix = ""

    if 'query_proj' in trainable or not quantize_only_trainable:
        groups.extend(get_proj_groups(getattr(self_attn, f"query{suffix}"), layer_n, prefix + f".attention.self.query{suffix}", model_name, hidden_act, True, prune_rank))

    if 'key_proj' in trainable or not quantize_only_trainable:
        groups.extend(get_proj_groups(
            getattr(self_attn, f"key{suffix}"), layer_n, prefix + f".attention.self.key{suffix}", model_name, hidden_act, True, prune_rank)
        )
    if 'value_proj' in trainable or not quantize_only_trainable:
        groups.extend(get_proj_groups(
            getattr(self_attn, f"value{suffix}"), layer_n, prefix + f".attention.self.value{suffix}", model_name, hidden_act, True, prune_rank)
        )
    if not quantize_only_trainable:
        names = ["attention_scores"]
        attrs = [self_attn.attention_scores_quantizer]

        if model_name == "deberta":
            names.extend(["p2c", "c2p"])
            attrs.extend([self_attn.p2c_quantizer, self_attn.c2p_quantizer])

        for name, attr in zip(names, attrs):
            groups.append(((self_template.format(name=name),), [attr]))
        groups.append(((self_template.format(name="context_activation"), out_template.format(name="weight")),
                       [self_attn.context_activation_quantizer, attn_layer.output.dense.weight_quantizer]))
    return groups


def get_transformers_layer_groups(layer: nn.Module, layer_n: int, prefix: str, model_name: str,
                                  hidden_act: torch.tensor = None, quantize_only_trainable: bool = True,
                                  trainable: list[str] = None, prune_rank: bool = False):
    """
    Groups model layer quantizers.

    :param layer: transformer layer.
    :param layer_n: number of transformer layer (from 0 to config.num_hidden_layers - 1).
    :param prefix: string that represents the parent class.
    :param model_name: name of the model.
    :param hidden_act: output of the previous layer (input of the current layer).
    :param quantize_only_trainable: if used, method applied only to LoRA blocks.
    :param trainable: modules to apply LoRA to.
    :param prune_rank: if rank is pruned.

    :return: grouped quantizers.
    """
    q_template = prefix + ".layer." + str(layer_n) + ".{name1}.dense.{name2}_quantizer"
    out_template = prefix + ".layer." + str(layer_n) + ".attention.output.dense.{name}_quantizer"

    groups = get_attention_groups(layer.attention, layer_n, prefix + f".layer.{layer_n}", model_name, hidden_act, quantize_only_trainable, trainable, prune_rank)
    if not quantize_only_trainable:
        groups.append(((out_template.format(name="activation"), q_template.format(name1="intermediate", name2="weight")),
                       [layer.attention.output.dense.activation_quantizer, layer.intermediate.dense.weight_quantizer]))
        groups.append(((q_template.format(name1="intermediate", name2="activation"), q_template.format(name1="output", name2="weight")),
                       [layer.intermediate.dense.weight_quantizer, layer.output.dense.weight_quantizer]))
    return groups


def get_transformers_groups(model: nn.Module, config, model_name, quantize_only_trainable: bool = True,
                            trainable: list[str] = None, prune_rank: bool = False):
    """
    Groups given model.

    :param model: model.
    :param config: transformers config.
    :param model_name: name of the model.
    :param quantize_only_trainable: if used, method applied only to LoRA blocks.
    :param trainable: modules to apply LoRA to.
    :param prune_rank: if rank is pruned.

    :return: grouped quantizers.
    """
    groups = []
    prefix = f"{model_name}.encoder"

    if model_name == 'deberta':
        module_name = "DebertaV2ForSequenceClassification."
    elif model_name == 'roberta':
        module_name = "RobertaForSequenceClassification."
    else:
        raise ValueError("Logging for this model not supported.")

    if hasattr(model, model_name):
        encoder = getattr(model, model_name).encoder
        prefix = module_name + prefix
    else:
        encoder = model.encoder

    hidden_act = None

    for i in range(config.num_hidden_layers):
        groups.extend(get_transformers_layer_groups(encoder.layer[i], i, prefix, model_name, hidden_act,
                                                    quantize_only_trainable, trainable, prune_rank))
        if not quantize_only_trainable:
            hidden_act = encoder.layer[i].output.dense.activation_quantizer

    if hasattr(model, model_name):
        if not quantize_only_trainable:
            act = prefix + f".layer.{config.num_hidden_layers - 1}.output.dense.activation_quantizer"
        q_template = module_name + "{name1}.{name2}_quantizer"
        if not quantize_only_trainable:
            if model_name == "deberta":
                groups.append(((act, q_template.format(name1="pooler.dense", name2="weight"),),
                               [hidden_act, model.pooler.dense.weight_quantizer]))
            elif model_name == "roberta":
                groups.append(((act, q_template.format(name1="classifier.dense", name2="weight"),),
                               [hidden_act, model.classifier.dense.weight_quantizer]))
        else:
            if hasattr(model.pooler, "weight_quantizer"):
                if model_name == "deberta":
                    groups.append(((q_template.format(name1="pooler.dense", name2="weight"),),
                                   [model.pooler.dense.weight_quantizer]))
                elif model_name == "roberta":
                    groups.append(((q_template.format(name1="classifier.dense", name2="weight"),),
                                   [model.classifier.dense.weight_quantizer]))

        if hasattr(model.classifier, "weight_quantizer"):
            if model_name == "deberta":
                groups.append(((q_template.format(name1="pooler.dense", name2='activation'),
                                q_template.format(name1="classifier", name2="weight"),),
                               [model.pooler.dense.activation_quantizer, model.classifier.weight_quantizer]))
                groups.append(((q_template.format(name1="classifier", name2="activation"),),
                               [model.classifier.activation_quantizer]))
            elif model_name == "roberta":
                groups.append(((q_template.format(name1="classifier.dense", name2='activation'),
                                q_template.format(name1="classifier.out_proj", name2="weight"),),
                               [model.classifier.dense.activation_quantizer, model.classifier.out_proj.weight_quantizer]))
                groups.append(((q_template.format(name1="classifier.out_proj", name2="activation"),),
                               [model.classifier.out_proj.activation_quantizer]))

        return groups


task_2_metrics = {'qqp': ['eval_accuracy', 'eval_f1'], 'rte': ['eval_accuracy', 'eval_f1'], 'stsb': ['eval_pearson', 'eval_spearmanr'], 'cola': ['eval_matthews_correlation'], 'mrpc': ['eval_accuracy', 'eval_f1']}


class BloraCallback(TrainerCallback):
    def __init__(self, task_name, quantizers, relevant_quantizers, prune_rank, rank=0, use_wandb=True):
        self.metrics = task_2_metrics.get(task_name, ['eval_accuracy'])
        self.quantizers = quantizers
        self.relevant_quantizers = relevant_quantizers
        self.prune_rank = prune_rank
        self.rank = rank
        self.use_wandb = use_wandb
        pretty_print_quantization(
            self.relevant_quantizers, "bayesian_bits", logfile=None, prune_only=False, prune_rank=prune_rank,
            log_wandb=self.use_wandb, max_rank=rank
        )

    def on_evaluate(self, args, state, control, **kwargs):
        metrics = kwargs['metrics']

        pretty_print_quantization(
            self.relevant_quantizers, "bayesian_bits", logfile=None, prune_only=False, prune_rank=self.prune_rank,
            max_rank=self.rank, log_wandb=self.use_wandb
        )

        if self.use_wandb:
            wandb_log = {}
            wandb_log["train/epoch"] = state.epoch

            for m in self.metrics:
                if m in metrics.keys():
                    wandb_log[f"train/{m.split('_')[-1]}"] = metrics[m]
            wandb_log["train/eval_loss"] = metrics["eval_loss"]
            wandb.log(wandb_log)


# =============================================================================
# The following functions are needed for logging results.
# They are taken from https://github.com/Qualcomm-AI-research/BayesianBits.
# Some of them were updated to be compatible with transformers.
# =============================================================================


def get_relevant_quantizers_from_groups(quantizer_groups):
    result = []
    for qg, quantizers in quantizer_groups:
        result += list(zip(qg, quantizers))
    return result


def print_and_log(*s, logfile=None, **kwargs):
    print(*s, **kwargs)
    if logfile:
        print(*s, **kwargs, file=open(logfile, "a"))
        
        
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def get_bw(quantizer, prune_only=False):
    if prune_only:
        if quantizer.quantizer is None or not (
            quantizer.quantizer.fixed_8bit or quantizer.quantizer.fixed_4bit
        ):
            return 16
    if "bayesian_bits" in quantizer.method:
        assert quantizer.quantizer is not None
        if quantizer.quantizer.fixed_8bit:
            return 8
        elif quantizer.quantizer.fixed_4bit:
            return 4

        train = quantizer.quantizer.training
        if train:
            quantizer.quantizer.eval()

        fix_type = lambda g: int(g.item()) if isinstance(g, torch.Tensor) else int(g)
        q4, q8, q16, q32 = [fix_type(g) for g in quantizer.quantizer.get_gates()[1:]]
        n = 1 + q4 + q4 * (q8 + q8 * (q16 + q16 * q32))

        if train:
            quantizer.quantizer.train()

        return int(2 ** n)
    else:
        return quantizer.n_bits


def pretty_print_quantization(
        relevant_quantizers, method, logfile=None, prune_only=False, log_wandb=False, prune_rank=False, max_rank=0
):
    apply_sigmoid = method == "bayesian_bits"
    n_bits_txt = ""
    max_len = max([len(nm) for nm, _ in relevant_quantizers])

    template = (
        "| {{:<{ml}}} | {{:<7}} || "
        "{{:>{ns}.4f}} | {{:>{ns}.4f}} | {{:>{ns}.4f}} "
        "| {{:>{ns}.4f}} |"
    )
    if prune_rank:
        template += "| {{:>{ns}}} |"
        rank_template = " | ".join(["{{:>{ns}.4f}}" for _ in range(max_rank)]) + "\n"
        rank_template = rank_template.format(ns=16)
        rank_txt = ""

    template += "| {{x_min:>8.4f}} | {{x_max:>8.4f}} |\n"
    template = template.format(ml=max_len, ns=6 + 2 * int(not apply_sigmoid), lpi=8)

    dummy_zeros = [0] * 4 * (1 + (method == "bayesian_bits"))
    if prune_rank:
        dummy_line = template.format("a", 8, *dummy_zeros, "", x_min=0.0, x_max=0.0)
        dummy_rank_line = rank_template.format(*[0 for _ in range(max_rank)])
        rank_hline = "|" + "-" * (len(dummy_rank_line) - 3) + "|"
    else:
        dummy_line = template.format("a", 8, *dummy_zeros, x_min=0.0, x_max=0.0)
    hline = "|" + "-" * (len(dummy_line) - 3) + "|"

    if log_wandb:
        wandb_log = {}

    for name, quantizer in relevant_quantizers:
        bw = get_bw(quantizer, prune_only)
        if not prune_only and method is not None:
            gams = [
                getattr(quantizer.quantizer, "gamma_{}".format(2 ** i)).item()
                for i in range(2, 6)
            ]
        else:
            gams = [0] * 4
        if apply_sigmoid:
            gams = [sigmoid(g) for g in gams]
        if quantizer.quantizer is not None:
            x_min, x_max = (
                quantizer.quantizer.x_min.item(),
                quantizer.quantizer.x_max.item(),
            )
        else:
            x_min = x_max = float("nan")

        if prune_rank and quantizer.quantizer.prune_rank:
            rank = quantizer.quantizer.curr_rank
        else:
            rank = 0

        if prune_rank:
            n_bits_txt += template.format(
                name,
                int(np.log2(bw)) * "*",
                *gams,
                rank,
                x_min=x_min,
                x_max=x_max
            )
            if quantizer.quantizer.gamma_2 is not None:
                rank_gammas = [1.0] + [sigmoid(g.item()) for g in quantizer.quantizer.gamma_2]
                rank_txt += rank_template.format(*rank_gammas)
        else:
            n_bits_txt += template.format(
                name,
                int(np.log2(bw)) * "*",
                *gams,
                x_min=x_min,
                x_max=x_max
            )

        if log_wandb:
            g = 4
            for i in range(len(gams)):
                wandb_log[f"train/{name}/gamma_{g}"] = gams[i]
                g *= 2

    print_and_log(hline.replace("|", "-"), logfile=logfile)

    # Make header: | Quantizer | ln(B) || g2 | ... | g32 | x_min | x_max |
    # make title elements for header, put in list:
    hs = ["g"]
    header = ["Quantizer", "log2(B)"] + [
        h + str(2 ** i) for h in hs for i in range(2, 6)
    ]
    if prune_rank:
        header.append('Rank')
        rank_header = [f"Gamma Rank {i + 1}" for i in range(max_rank)]
    # format header with title elements:
    print_and_log(
        template.replace(".4f", "")
        .replace(".2f", "")
        .format(*header, x_min="x_min", x_max="x_max"),
        end="",
        logfile=logfile,
    )
    print_and_log(hline, logfile=logfile)

    # Add the rest of the text
    print_and_log(n_bits_txt, end="", logfile=logfile)
    print_and_log(hline.replace("|", "-"), logfile=logfile)

    if prune_rank:
        print_and_log(
            rank_template.replace(".4f", "")
            .replace(".2f", "")
            .format(*rank_header),
            end="",
            logfile=logfile,
        )
        print_and_log(rank_hline, logfile=logfile)

        print_and_log(rank_txt, end="", logfile=logfile)
        print_and_log(rank_hline.replace("|", "-"), logfile=logfile)

    print_and_log(logfile=logfile)

    if log_wandb:
        return wandb_log
