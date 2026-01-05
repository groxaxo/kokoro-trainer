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
    def __init__(self, lang_code: str = "a"):
        """
        Initialize SpeechGenerator with language support
        
        Args:
            lang_code: Language code for Kokoro pipeline
                      'a' = Auto-detect (default)
                      'es' = Spanish (if supported by model)
                      'en' = English
        """
        if USING_ONNX:
            raise ImportError(
                "The 'kokoro' package is required but not installed.\n"
                "Please install it with: pip install kokoro\n"
                "or use uv sync to install all dependencies."
            )
        
        suppressWarnings()
        self.lang_code = lang_code
        self.pipeline = KPipeline(lang_code=lang_code, repo_id='hexgrad/Kokoro-82M')

    def generate_audio(self, text: str, voice: torch.Tensor, speed: float = 1.0) -> np.typing.NDArray[np.float32]:
        """
        Generate audio from text and voice tensor
        
        Args:
            text: Text to synthesize
            voice: Voice tensor (style vector)
            speed: Speech speed multiplier (1.0 = normal)
                   For Spanish: 0.9-1.0 recommended for clarity
                   
        Returns:
            Audio array (numpy float32, 24kHz)
        """
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
