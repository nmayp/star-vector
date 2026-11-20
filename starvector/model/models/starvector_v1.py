import torch
import torch.nn as nn
from starvector.model.models.starvector_base import StarVectorBase
from transformers import AutoProcessor

class StarVectorStarCoder(StarVectorBase):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)

        name_or_path = getattr(config, "_name_or_path", "")
        if name_or_path:
            try:
                self.processor = AutoProcessor.from_pretrained(name_or_path)
            except Exception:
                self.processor = None
        else:
            self.processor = None

    def _get_svg_transformer(self, config, **kwargs):
        from starvector.model.llm.starcoder import StarCoderModel # This uses StarCoder (V1)
        return StarCoderModel(config, **kwargs)

    def get_fsdp_wrapping_policy(self):
        """V1 specific FSDP wrapping policy"""
        from functools import partial
        from torch.distributed.fsdp.wrap import _module_wrap_policy, _or_policy
        from starvector.model.image_encoder.image_encoder import ImageEncoder
        from starvector.model.adapters.adapter import Adapter

        policies = []
        if self.use_image_encoder():
            policies.append(partial(_module_wrap_policy, module_classes={ImageEncoder}))
            policies.append(partial(_module_wrap_policy, module_classes={Adapter}))
        return partial(_or_policy, policies=policies) if policies else None

    def _get_embeddings(self, input_ids):
        """V1 specific embedding method"""
        return self.svg_transformer.transformer.transformer.wte(input_ids)

    def _get_svg_text(self, svg_list):
        """V1 specific SVG text preparation"""
        return [t + self.svg_transformer.tokenizer.eos_token for t in svg_list]
