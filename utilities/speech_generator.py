import warnings

import numpy as np
import torch

# Try to import the kokoro package
try:
    from kokoro import KPipeline
    USING_ONNX = False
except ImportError:
    USING_ONNX = True
    print("⚠️  kokoro package not found. Some features may be limited.")
    print("   Install with: pip install kokoro")


class SpeechGenerator:
    def __init__(self):
        if USING_ONNX:
            raise ImportError(
                "The 'kokoro' package is required but not installed.\n"
                "Please install it with: pip install kokoro\n"
                "or use uv sync to install all dependencies."
            )
        
        suppressWarnings()
        self.pipeline = KPipeline(lang_code="a", repo_id='hexgrad/Kokoro-82M')

    def generate_audio(self, text: str, voice: torch.Tensor, speed: float = 1.0) -> np.typing.NDArray[np.float32]:
        generator = self.pipeline(text, voice, speed)
        audio = []
        for gs, ps, chunk in generator:
            audio.append(chunk)
        return np.concatenate(audio)

def suppressWarnings():
    # Suppress all these warnings showing up from libraries cluttering the console
    warnings.filterwarnings(
        "ignore",
        message=".*RNN module weights are not part of single contiguous chunk of memory.*",
        category=UserWarning,
    )
    warnings.filterwarnings(
        "ignore", message=".*is deprecated in favor of*", category=FutureWarning
    )
    warnings.filterwarnings(
        "ignore",
        message=".*dropout option adds dropout after all but last recurrent layer*",
        category=UserWarning,
    )
