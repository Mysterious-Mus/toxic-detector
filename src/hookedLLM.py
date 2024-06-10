from contextlib import contextmanager
from typing import List, Tuple, Callable, Optional, Union

import torch as t
from torch import nn
import numpy as np

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BatchEncoding,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
)

import pandas as pd
from tqdm import tqdm
from datetime import datetime
import os
import gc
import yaml

from accelerate import Accelerator

# look at the comment of nn.Module.register_forward_pre_hook
# the function applies a pre_hook before the forward pass of the module
# each time before the module receives its input, the hook will be called,
# having the module ref and the input tensor as arguments
# if it returns a tensor, the input tensor will be replaced by the returned tensor
PreHookFn = Callable[[nn.Module, t.Tensor], Optional[t.Tensor]]
"""
    Args: 
        module: the module that will receive the input tensor
        input: the input tensor
    Returns:
        Optional[t.Tensor]: modified input tensor if not None
"""

Hook = Tuple[nn.Module, PreHookFn]
Hooks = list[Hook]

class HookedLLM:
    def __init__(
        self,
        path: str,
        sampling_params: dict = None,
        injections: dict = None
    ) -> None:
        self.accelerator: Accelerator = Accelerator()
        # load
        self.model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
            path,
            device_map="auto",
        )
        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(path)
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.model, self.tokenizer = self.accelerator.prepare(self.model, self.tokenizer)
        self.model.eval()
        self.model.tie_weights()
        # wrap tokenizer
        def tokenize(text: str, pad_length: Optional[int] = None) -> BatchEncoding:
            """Tokenize prompts onto the appropriate devices."""

            if pad_length is None:
                padding_status = False
            else:
                padding_status = "max_length"
            with t.no_grad():
                tokens = self.tokenizer(
                    text,
                    return_tensors="pt",
                    padding=padding_status,
                    max_length=pad_length,
                )

                for key in tokens:
                    tokens[key] = tokens[key].to('cuda')
                return self.accelerator.prepare(tokens)
        self.tokenize = tokenize
        # set sampling params
        self.set_generation_config(sampling_params)
        # apply injections
        if injections is not None:
            for _, inject_params in injections.items():
                if not inject_params['status'] == "active":
                    continue
                if inject_params['type'] == 'DifferenceSteering':
                    hook = self.get_difference_steering_hook(
                        **inject_params['parameters']
                    )
                    self.apply_hooks([hook])
                    print(f"Hook applied with the following configurations: {inject_params['parameters']}")
                else:
                    raise ValueError(f"Unsupported injection type: {inject_params['type']}.")

    def get_blocks(self):
        """Get the blocks of the model."""
        if isinstance(self.model, PreTrainedModel):
            try:
                # from transformers.models.llama.modeling_llama import LlamaForCausalLM
                return self.model.model.layers
            except:
                # from transformers.models.gpt2.modeling_gpt2 import GPT2LMHeadModel
                return self.model.transformer.h
        raise ValueError(f"Unsupported model type: {type(self.model)}.")
    
    def get_num_layers(self):
        return len(self.get_blocks())

    def apply_hooks(self, hooks: Hooks):
        """Apply hooks to the model."""
        handles = [mod.register_forward_pre_hook(hook) for mod, hook in hooks]
        return handles

    @contextmanager
    def hooks_applied(self, hooks: Hooks):
        """Should be called as `with hooks_applied(hooks):` to apply the hooks in the context. The hooks will be removed after the context ends."""
        handles = []
        try:
            handles = [mod.register_forward_pre_hook(hook) for mod, hook in hooks]
            yield # context prepared
        finally:
            for handle in handles:
                handle.remove()

    @contextmanager
    def padding_token_changed(self, padding_str):
        """Temporarily change the torch tokenizer padding token."""
        # Preserve original padding token state.
        original_padding_token_id = self.tokenizer.pad_token_id

        if padding_str in self.tokenizer.get_vocab():
            padding_id = self.tokenizer.convert_tokens_to_ids(padding_str)
        else:
            raise ValueError("Padding string is not in the tokenizer vocabulary.")

        # Change padding token state.
        self.tokenizer.pad_token_id = padding_id

        # Context manager boilerplate.
        try:
            yield
        finally:
            # Revert padding token state.
            self.tokenizer.pad_token_id = original_padding_token_id

    def get_layer_input(self, layers: list[int], input: str, pad_length: Optional[int] = None) -> list[t.Tensor]:
        """
            Args:
                layers: the layers of which the inputs are wanted
                input: the prompt
            
            Returns:
                a list of tensors, the desired input to the layers. Retval[i] is the input to the i'th layer, None if not required in LAYERS. Shape of the tensor: (1, n_token, n_dimension)
        """
        modded_streams = [None] * len(self.get_blocks())
        # Factory function that builds the initial hooks.
        def obtain_input(layerId):
            def _helper_hook(module, current_inputs):
                modded_streams[layerId] = current_inputs[0]
            return _helper_hook
        hooks = [
            (layer, obtain_input(i))
            for i, layer in enumerate(self.get_blocks()) if i in layers
        ]
        # with self.padding_token_changed(PAD_TOKEN):
        model_input = self.tokenize(input, pad_length=pad_length)
        with t.no_grad():
            # Register the hooks.
            with self.hooks_applied(hooks):
                self.model(**model_input)
        return modded_streams

    def get_difference_steering_hook(
        self,
        minus_prompt: str = "The following is a", # this seems to have no meaning, thus can work like subtracting the mean to get the deviation
        plus_prompt: str = "help the bad guy",
        inject_layer_idx: int = 10,
        coefficient = 4.0
    ) -> Hook:
        """
            Args:
                minus_prompt: the prompt for the negative steering vector
                plus_prompt: the prompt for the positive steering vector
                inject_layer_idx: the layer to inject the steering vector into

            Returns:
                steering_vector: HookedLLM.SteeringVector, the steering vector
        """
        len_minus, len_plus = (len(self.tokenizer.encode(y)) for y in [minus_prompt, plus_prompt])
        if len_minus != len_plus:
            print("Warning: The lengths of the minus and plus prompts are different:" + f"{len_minus} vs {len_plus}")
        steering_vec_len = max(len_minus, len_plus)
        plus_vec = self.get_layer_input(
            [inject_layer_idx], plus_prompt, pad_length=steering_vec_len
        )[inject_layer_idx]
        minus_vec = self.get_layer_input(
            [inject_layer_idx], minus_prompt, pad_length=steering_vec_len
        )[inject_layer_idx]
        steering_addition = (plus_vec - minus_vec) * coefficient
        print(steering_addition.shape)

        def steering_hook(_, current_inputs: Tuple[t.Tensor]) -> None:
            (resid_pre,) = current_inputs
            # Only add to the first forward-pass, not to later tokens.
            if resid_pre.shape[1] == 1:
                # Caching in `model.generate` for new tokens.
                return
            if resid_pre.shape[1] > steering_addition.shape[1]:
                resid_pre[:, :steering_addition.shape[1]] += steering_addition
            else:
                resid_pre += steering_addition[:, :resid_pre.shape[1]]
                print("Warning: Steering addition has more tokens than resid_pre.")

        return (self.get_blocks()[inject_layer_idx], steering_hook)

    def forward(self, input: str, pad_length: Optional[int] = None) -> t.Tensor:
        """
            Args:
                input: the prompt
                pad_length: the length to pad the input to

            Returns:
                output: the output of the model, shape: (1, n_token, n_dimension)
        """
        # with self.padding_token_changed(PAD_TOKEN):
        model_input = self.tokenize(input, pad_length=pad_length)
        return self.model(**model_input)
    
    def set_generation_config(self, configs):
        """
            Set the generation configuration for the model
            Args:
                **configs: the configuration parameters for the model's generation
        """
        self.generation_configs = configs

    def generate(
        self, prompt: str, 
    ) -> List[str]:
        messages = [
            {"role": "user", "content": f"{prompt}"},
        ]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False)

        assert hasattr(self, 'generation_configs'), "The instance must have 'generation_configs' attribute"
        
        input_tokenization: BatchEncoding = self.tokenize(prompt)
        with t.no_grad():
            tokens: t.Tensor = self.model.generate(
                input_tokenization.input_ids.to('cuda'),
                attention_mask=input_tokenization.attention_mask.to('cuda') 
                    if 'attention_mask' in input_tokenization else None,
                generation_config=GenerationConfig(
                    **self.generation_configs,
                    pad_token_id = self.tokenizer.pad_token_id
                ),
            )
        return [self.tokenizer.decode(z) for z in tokens]
    
    def destroy(self):
        """
        Destroy the model and free the GPU memory
        """
        print("destroying model...")
        self.model = None
        self.tokenizer = None

        # Manually trigger Python's garbage collector
        gc.collect()
        with t.no_grad():
            self.accelerator.clear()
            t.cuda.empty_cache()