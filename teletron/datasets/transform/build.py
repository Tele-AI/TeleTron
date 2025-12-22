from teletron.datasets.registry import Registry, build_module
from .prompt_transform import (
    PromptToClipEmbedding,
    PromptToTransformerEmbedding,
    PromptGenerator,
)
from .video_transform import (
    SampleImages, 
    SampleImageVideo,
    GenerateRefImages, 
    GenerateFirstRefImage,
    GenerateRefImagesWithMask, 
    GenerateRawFirstRefImage,
    GenerateRawFirstLastRefImage,
    GenerateRefImagesWithTimeMask,
    SampleDynamicFPSVideo,
    SampleWholeVideo
)
from .formatting import PackInputs

TRANSFORMS = Registry()
TRANSFORMS.register_module(SampleDynamicFPSVideo)
TRANSFORMS.register_module(SampleWholeVideo)
TRANSFORMS.register_module(PromptToClipEmbedding)
TRANSFORMS.register_module(PromptToTransformerEmbedding)
TRANSFORMS.register_module(PromptGenerator)
TRANSFORMS.register_module(SampleImages)
TRANSFORMS.register_module(SampleImageVideo)
TRANSFORMS.register_module(PackInputs)
TRANSFORMS.register_module(GenerateRefImages)
TRANSFORMS.register_module(GenerateFirstRefImage)
TRANSFORMS.register_module(GenerateRefImagesWithMask)
TRANSFORMS.register_module(GenerateRawFirstRefImage)
TRANSFORMS.register_module(GenerateRawFirstLastRefImage)
TRANSFORMS.register_module(GenerateRefImagesWithTimeMask)


def build_transform(params_or_type, *args, **kwargs):
    return build_module(TRANSFORMS, params_or_type, *args, **kwargs)
