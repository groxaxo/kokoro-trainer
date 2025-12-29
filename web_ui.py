"""
Gradio 6 Web UI for KVoiceWalk
A comprehensive interface for voice cloning, testing, and transcription
"""

import os
import tempfile
from pathlib import Path
from typing import Optional, Tuple
import argparse

import gradio as gr
import numpy as np
import soundfile as sf
import torch

from utilities.audio_processor import Transcriber, convert_to_wav_mono_24k
from utilities.kvoicewalk import KVoiceWalk
from utilities.speech_generator import SpeechGenerator
from utilities.pytorch_sanitizer import load_voice_safely


class WebUI:
    def __init__(self):
        # Lazy initialization - only create generators when needed
        self._speech_generator = None
        self.transcriber = None
    
    @property
    def speech_generator(self):
        """Lazy load speech generator only when needed"""
        if self._speech_generator is None:
            self._speech_generator = SpeechGenerator()
        return self._speech_generator
        
    def get_transcriber(self):
        """Lazy load transcriber only when needed"""
        if self.transcriber is None:
            self.transcriber = Transcriber()
        return self.transcriber
    
    def test_voice(
        self, 
        voice_file: Optional[str], 
        text: str,
        speed: float = 1.0
    ) -> Tuple[Optional[str], str]:
        """Test a voice file with given text"""
        try:
            if not voice_file:
                return None, "❌ Please upload a voice file (.pt)"
            
            if not text:
                return None, "❌ Please enter text to synthesize"
            
            # Load voice
            voice = load_voice_safely(voice_file, auto_allow_unsafe=True)
            if voice is None:
                return None, "❌ Failed to load voice file"
            
            # Generate audio
            audio = self.speech_generator.generate_audio(text, voice, speed=speed)
            
            # Save to temporary file
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
            sf.write(temp_file.name, audio, 24000)
            
            return temp_file.name, "✅ Voice generated successfully!"
            
        except Exception as e:
            return None, f"❌ Error: {str(e)}"
    
    def transcribe_audio(self, audio_file: Optional[str]) -> Tuple[str, str]:
        """Transcribe audio file to text"""
        try:
            if not audio_file:
                return "", "❌ Please upload an audio file"
            
            transcriber = self.get_transcriber()
            audio_path = Path(audio_file)
            
            # Convert to proper format if needed
            audio_path = convert_to_wav_mono_24k(audio_path)
            
            # Transcribe
            transcription = transcriber.transcribe(audio_path=audio_path)
            
            return transcription, "✅ Transcription completed!"
            
        except Exception as e:
            return "", f"❌ Error: {str(e)}"
    
    def clone_voice(
        self,
        target_audio: Optional[str],
        target_text: str,
        other_text: str,
        voice_folder: str,
        population_limit: int,
        step_limit: int,
        interpolate_start: bool,
        starting_voice: Optional[str],
        output_name: str,
        progress=gr.Progress()
    ) -> Tuple[Optional[str], Optional[str], str]:
        """Main voice cloning function"""
        try:
            if not target_audio:
                return None, None, "❌ Please upload target audio file"
            
            if not target_text:
                return None, None, "❌ Please enter target text"
            
            if not os.path.exists(voice_folder):
                return None, None, f"❌ Voice folder not found: {voice_folder}"
            
            progress(0, desc="Converting audio format...")
            # Convert audio to proper format
            target_audio_path = Path(target_audio)
            target_audio_path = convert_to_wav_mono_24k(target_audio_path)
            
            progress(0.1, desc="Initializing voice cloning...")
            # Initialize KVoiceWalk
            kvw = KVoiceWalk(
                target_audio_path,
                target_text,
                other_text,
                voice_folder,
                interpolate_start,
                population_limit,
                starting_voice,
                output_name
            )
            
            progress(0.2, desc="Running voice cloning (this may take a while)...")
            # Run random walk
            kvw.random_walk(step_limit)
            
            progress(1.0, desc="Complete!")
            
            # Find the best generated voice
            results_dir = None
            for item in os.listdir("./out"):
                if item.startswith(output_name) and target_audio_path.stem in item:
                    results_dir = Path("./out") / item
                    break
            
            if results_dir and results_dir.exists():
                # Get the best files (highest score)
                wav_files = list(results_dir.glob("*.wav"))
                pt_files = list(results_dir.glob("*.pt"))
                
                if wav_files:
                    # Sort by score (in filename)
                    best_wav = max(wav_files, key=lambda x: float(x.stem.split('_')[-3]))
                    best_pt = None
                    
                    # Find corresponding .pt file
                    for pt in pt_files:
                        if pt.stem == best_wav.stem:
                            best_pt = pt
                            break
                    
                    return (
                        str(best_wav),
                        str(best_pt) if best_pt else None,
                        f"✅ Voice cloning completed!\n📁 Results in: {results_dir}\n🎵 Best audio: {best_wav.name}"
                    )
            
            return None, None, "✅ Voice cloning completed but no results found"
            
        except Exception as e:
            import traceback
            return None, None, f"❌ Error: {str(e)}\n{traceback.format_exc()}"
    
    def quick_clone_with_transcription(
        self,
        target_audio: Optional[str],
        voice_folder: str,
        population_limit: int,
        step_limit: int,
        interpolate_start: bool,
        output_name: str,
        progress=gr.Progress()
    ) -> Tuple[str, Optional[str], Optional[str], str]:
        """Clone voice with automatic transcription"""
        try:
            if not target_audio:
                return "", None, None, "❌ Please upload target audio file"
            
            progress(0, desc="Transcribing audio...")
            # Transcribe first
            transcription, msg = self.transcribe_audio(target_audio)
            
            if not transcription:
                return "", None, None, f"❌ Transcription failed: {msg}"
            
            progress(0.3, desc="Starting voice cloning...")
            # Use default other_text
            other_text = "If you mix vinegar, baking soda, and a bit of dish soap in a tall cylinder, the resulting eruption is both a visual and tactile delight, often used in classrooms to simulate volcanic activity on a miniature scale."
            
            # Clone voice
            audio, voice_pt, clone_msg = self.clone_voice(
                target_audio,
                transcription,
                other_text,
                voice_folder,
                population_limit,
                step_limit,
                interpolate_start,
                None,
                output_name,
                progress
            )
            
            return transcription, audio, voice_pt, f"✅ Transcription:\n{transcription}\n\n{clone_msg}"
            
        except Exception as e:
            import traceback
            return "", None, None, f"❌ Error: {str(e)}\n{traceback.format_exc()}"


def create_interface():
    """Create the Gradio interface"""
    ui = WebUI()
    
    with gr.Blocks(
        title="KVoiceWalk - Voice Cloning Tool",
        theme=gr.themes.Soft()
    ) as app:
        gr.Markdown(
            """
            # 🎙️ KVoiceWalk - Voice Cloning Tool
            
            Clone voices using random walk algorithm and Kokoro TTS.
            Create new voice styles that match target audio samples.
            """
        )
        
        with gr.Tabs():
            # Tab 1: Voice Testing
            with gr.Tab("🎵 Test Voice"):
                gr.Markdown("### Test a voice file with custom text")
                
                with gr.Row():
                    with gr.Column():
                        test_voice_file = gr.File(
                            label="Voice File (.pt)",
                            file_types=[".pt"]
                        )
                        test_text = gr.Textbox(
                            label="Text to Synthesize",
                            placeholder="Enter the text you want to hear...",
                            lines=3,
                            value="The old lighthouse keeper never imagined that one day he'd be guiding ships from the comfort of his living room."
                        )
                        test_speed = gr.Slider(
                            minimum=0.5,
                            maximum=2.0,
                            value=1.0,
                            step=0.1,
                            label="Speech Speed"
                        )
                        test_btn = gr.Button("🎙️ Generate Speech", variant="primary")
                    
                    with gr.Column():
                        test_output = gr.Audio(label="Generated Audio", type="filepath")
                        test_status = gr.Textbox(label="Status", lines=2)
                
                test_btn.click(
                    fn=ui.test_voice,
                    inputs=[test_voice_file, test_text, test_speed],
                    outputs=[test_output, test_status]
                )
            
            # Tab 2: Voice Cloning (Manual)
            with gr.Tab("🔬 Clone Voice (Advanced)"):
                gr.Markdown("### Clone a voice with full control over parameters")
                
                with gr.Row():
                    with gr.Column():
                        clone_target_audio = gr.Audio(
                            label="Target Audio (20-30 seconds recommended)",
                            type="filepath"
                        )
                        clone_target_text = gr.Textbox(
                            label="Target Text (what is said in the audio)",
                            placeholder="Enter the exact text spoken in the audio...",
                            lines=3
                        )
                        clone_other_text = gr.Textbox(
                            label="Other Text (for self-similarity testing)",
                            lines=3,
                            value="If you mix vinegar, baking soda, and a bit of dish soap in a tall cylinder, the resulting eruption is both a visual and tactile delight, often used in classrooms to simulate volcanic activity on a miniature scale."
                        )
                        clone_voice_folder = gr.Textbox(
                            label="Voice Folder Path",
                            value="./voices"
                        )
                        
                        with gr.Row():
                            clone_population = gr.Slider(
                                minimum=1,
                                maximum=50,
                                value=10,
                                step=1,
                                label="Population Limit"
                            )
                            clone_steps = gr.Slider(
                                minimum=100,
                                maximum=50000,
                                value=10000,
                                step=100,
                                label="Step Limit"
                            )
                        
                        clone_interpolate = gr.Checkbox(
                            label="Use Interpolation Start (slower but better)",
                            value=False
                        )
                        clone_starting_voice = gr.File(
                            label="Starting Voice (optional)",
                            file_types=[".pt"]
                        )
                        clone_output_name = gr.Textbox(
                            label="Output Name",
                            value="my_new_voice"
                        )
                        
                        clone_btn = gr.Button("🚀 Start Cloning", variant="primary")
                    
                    with gr.Column():
                        clone_audio_output = gr.Audio(label="Best Generated Audio", type="filepath")
                        clone_voice_output = gr.File(label="Best Voice File (.pt)")
                        clone_status = gr.Textbox(label="Status", lines=10)
                
                clone_btn.click(
                    fn=ui.clone_voice,
                    inputs=[
                        clone_target_audio,
                        clone_target_text,
                        clone_other_text,
                        clone_voice_folder,
                        clone_population,
                        clone_steps,
                        clone_interpolate,
                        clone_starting_voice,
                        clone_output_name
                    ],
                    outputs=[clone_audio_output, clone_voice_output, clone_status]
                )
            
            # Tab 3: Quick Clone with Transcription
            with gr.Tab("⚡ Quick Clone"):
                gr.Markdown("### Clone a voice with automatic transcription")
                
                with gr.Row():
                    with gr.Column():
                        quick_target_audio = gr.Audio(
                            label="Target Audio (20-30 seconds recommended)",
                            type="filepath"
                        )
                        quick_voice_folder = gr.Textbox(
                            label="Voice Folder Path",
                            value="./voices"
                        )
                        
                        with gr.Row():
                            quick_population = gr.Slider(
                                minimum=1,
                                maximum=50,
                                value=10,
                                step=1,
                                label="Population Limit"
                            )
                            quick_steps = gr.Slider(
                                minimum=100,
                                maximum=50000,
                                value=5000,
                                step=100,
                                label="Step Limit"
                            )
                        
                        quick_interpolate = gr.Checkbox(
                            label="Use Interpolation Start",
                            value=False
                        )
                        quick_output_name = gr.Textbox(
                            label="Output Name",
                            value="quick_voice"
                        )
                        
                        quick_btn = gr.Button("⚡ Quick Clone", variant="primary")
                    
                    with gr.Column():
                        quick_transcription = gr.Textbox(
                            label="Auto-Transcription",
                            lines=3
                        )
                        quick_audio_output = gr.Audio(label="Best Generated Audio", type="filepath")
                        quick_voice_output = gr.File(label="Best Voice File (.pt)")
                        quick_status = gr.Textbox(label="Status", lines=8)
                
                quick_btn.click(
                    fn=ui.quick_clone_with_transcription,
                    inputs=[
                        quick_target_audio,
                        quick_voice_folder,
                        quick_population,
                        quick_steps,
                        quick_interpolate,
                        quick_output_name
                    ],
                    outputs=[quick_transcription, quick_audio_output, quick_voice_output, quick_status]
                )
            
            # Tab 4: Transcription Only
            with gr.Tab("📝 Transcribe"):
                gr.Markdown("### Transcribe audio to text")
                
                with gr.Row():
                    with gr.Column():
                        transcribe_audio = gr.Audio(
                            label="Audio File",
                            type="filepath"
                        )
                        transcribe_btn = gr.Button("📝 Transcribe", variant="primary")
                    
                    with gr.Column():
                        transcribe_output = gr.Textbox(
                            label="Transcription",
                            lines=5
                        )
                        transcribe_status = gr.Textbox(label="Status", lines=2)
                
                transcribe_btn.click(
                    fn=ui.transcribe_audio,
                    inputs=[transcribe_audio],
                    outputs=[transcribe_output, transcribe_status]
                )
            
            # Tab 5: Help
            with gr.Tab("❓ Help"):
                gr.Markdown(
                    """
                    ## How to Use KVoiceWalk
                    
                    ### 🎵 Test Voice
                    - Upload a `.pt` voice file
                    - Enter text to synthesize
                    - Adjust speech speed
                    - Generate and listen to the result
                    
                    ### 🔬 Clone Voice (Advanced)
                    - Upload target audio (20-30 seconds of clean speech)
                    - Enter the exact text spoken in the audio
                    - Configure cloning parameters:
                      - **Population Limit**: Number of top voices to use (higher = more diverse)
                      - **Step Limit**: Number of random walk iterations (higher = better but slower)
                      - **Interpolation Start**: Find better starting point (slower but recommended)
                    - Wait for the process to complete
                    - Download the best voice file and audio
                    
                    ### ⚡ Quick Clone
                    - Upload target audio
                    - Audio is automatically transcribed
                    - Voice cloning starts automatically
                    - Faster workflow for quick results
                    
                    ### 📝 Transcribe
                    - Upload any audio file
                    - Get text transcription
                    - Use transcription for voice cloning
                    
                    ## Tips
                    - Use high-quality, clear audio for best results
                    - 20-30 seconds of single speaker audio works best
                    - Higher step limits produce better results but take longer
                    - Save your best voice files for future use
                    - Experiment with different population limits and interpolation settings
                    
                    ## Requirements
                    - Audio should be mono, 24kHz (auto-converted if not)
                    - Text should match what's spoken in the audio
                    - GPU recommended for faster processing
                    """
                )
        
        gr.Markdown(
            """
            ---
            **KVoiceWalk** - Voice cloning with random walk algorithm | 
            Built with [Kokoro TTS](https://github.com/hexgrad/kokoro) and [Gradio](https://gradio.app)
            """
        )
    
    return app


def main():
    parser = argparse.ArgumentParser(description="KVoiceWalk Web UI")
    parser.add_argument("--share", action="store_true", help="Create a public share link")
    parser.add_argument("--port", type=int, default=7860, help="Port to run the server on")
    parser.add_argument("--server-name", type=str, default="127.0.0.1", help="Server name/IP")
    args = parser.parse_args()
    
    app = create_interface()
    app.launch(
        share=args.share,
        server_name=args.server_name,
        server_port=args.port,
        show_error=True
    )


if __name__ == "__main__":
    main()
