import datetime
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
from tqdm import tqdm

from utilities.fitness_scorer import FitnessScorer
from utilities.initial_selector import InitialSelector
from utilities.path_router import OUT_DIR
from utilities.speech_generator import SpeechGenerator
from utilities.voice_generator import VoiceGenerator

# Import Spanish language utilities
try:
    from utilities.spanish_utils import SpanishTextNormalizer, SpanishVoiceScorer
    SPANISH_UTILS_AVAILABLE = True
except ImportError:
    SPANISH_UTILS_AVAILABLE = False

# Import CMA-ES if available
try:
    import cma
    CMA_AVAILABLE = True
except ImportError:
    CMA_AVAILABLE = False


def clear_gpu_memory():
    """Clear GPU cache to optimize memory usage"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# Constants for CMA-ES optimization
CMA_SHORT_TEXT_LENGTH = 200  # Character limit for text during optimization


class KVoiceWalk:
    def __init__(self, target_audio: Path, target_text: str, other_text: str, voice_folder: str,
                 interpolate_start: bool, population_limit: int, starting_voice: str, output_name: str,
                 use_super_seed: bool = False, lang_code: str = "a") -> None:
        try:
            self.target_audio = target_audio
            self.target_text = target_text
            self.other_text = other_text
            
            # Detect if we're working with Spanish text
            self.is_spanish_mode = False
            if SPANISH_UTILS_AVAILABLE:
                normalizer = SpanishTextNormalizer()
                is_spanish, confidence = normalizer.is_spanish_text(target_text)
                if is_spanish and confidence > 0.5:
                    self.is_spanish_mode = True
                    print(f"🇪🇸 Spanish language detected (confidence: {confidence:.2%})")
                    print("   Enabling Spanish-optimized training parameters...")
                    
                    # Get Spanish-specific recommendations
                    spanish_scorer = SpanishVoiceScorer()
                    recommendations = spanish_scorer.get_latin_american_recommendations(target_text)
                    
                    print("\n📋 Training Recommendations for Latin American Spanish:")
                    for tip in recommendations['quality_tips']:
                        print(f"   {tip}")
                    
                    if recommendations['training_params']:
                        print("\n⚙️  Recommended parameters:")
                        for param, value in recommendations['training_params'].items():
                            print(f"   --{param} {value}")
                    print()
            
            self.initial_selector = InitialSelector(str(target_audio), target_text, other_text,
                                                    voice_folder=voice_folder)
            voices: list[torch.Tensor] = []
            
            # Determine starting strategy
            if interpolate_start:
                voices = self.initial_selector.interpolate_search(population_limit)
            else:
                voices = self.initial_selector.top_performer_start(population_limit)
            
            # Use language-aware speech generator
            self.speech_generator = SpeechGenerator(lang_code=lang_code)
            self.fitness_scorer = FitnessScorer(str(target_audio))
            
            # Set target text for advanced scoring
            self.fitness_scorer.target_text = target_text
            
            # Enable Spanish mode in fitness scorer
            if self.is_spanish_mode:
                self.fitness_scorer.is_spanish_mode = True
            
            self.voice_generator = VoiceGenerator(voices, starting_voice)
            
            # Use super-seed if requested (average of top 5 voices)
            if use_super_seed and starting_voice is None:
                print("Using super-seed initialization (average of top 5 voices)")
                self.starting_voice = self.initial_selector.get_super_seed(top_k=5)
            else:
                # Either the mean or the supplied voice tensor
                self.starting_voice = self.voice_generator.starting_voice
            
            self.output_name = output_name
        except Exception as e:
            print(f"Error initializing KVoicewalk: {e}")

    def random_walk(self,step_limit: int):

        # Score Initial Voice
        best_voice = self.starting_voice
        best_results = self.score_voice(self.starting_voice)
        t = tqdm()
        t.write(f'Target Sim:{best_results["target_similarity"]:.3f}, Self Sim:{best_results["self_similarity"]:.3f}, Feature Sim:{best_results["feature_similarity"]:.2f}, Score:{best_results["score"]:.2f}')

        # Create Results Directory
        now = datetime.datetime.now()
        results_dir = Path(OUT_DIR / f'{self.output_name}_{self.target_audio.stem}_{now.strftime("%Y%m%d_%H%M%S")}')
        os.makedirs(results_dir, exist_ok=True)

        # Random Walk Loop

        for i in tqdm(range(step_limit)):
            # TODO: Expose to CLI
            diversity = random.uniform(0.01,0.15)
            voice = self.voice_generator.generate_voice(best_voice,diversity)

            # Early function return saves audio generation compute
            min_similarity = best_results["target_similarity"] * 0.98
            voice_results = self.score_voice(voice,min_similarity)

            # Set new winner if score is better
            if voice_results["score"] > best_results["score"]:
                best_results = voice_results
                best_voice = voice
                t.write(f'Step:{i:<4} Target Sim:{best_results["target_similarity"]:.3f} Self Sim:{best_results["self_similarity"]:.3f} Feature Sim:{best_results["feature_similarity"]:.3f} Score:{best_results["score"]:.2f} Diversity:{diversity:.2f}')
                # Save results so folks can listen
                torch.save(best_voice,
                           f'{results_dir}/{self.output_name}_{i}_{best_results["score"]:.2f}_{best_results["target_similarity"]:.2f}_{self.target_audio.stem}.pt')
                sf.write(
                    f'{results_dir}/{self.output_name}_{i}_{best_results["score"]:.2f}_{best_results["target_similarity"]:.2f}_{self.target_audio.stem}.wav',
                    best_results["audio"], 24000)
                # Clear GPU memory periodically
                if i % 100 == 0:
                    clear_gpu_memory()
                # TODO: Add config file for easy restarting runs from last save point

        # Print Final Results for Random Walk
        print(f"Random Walk Final Results for {self.output_name}")
        print(f"Duration: {t.format_dict['elapsed']}")
        # print(f"Best Voice: {best_voice}") #TODO: add best voice model name
        print(f"Best Score: {best_results['score']:.2f}_")
        print(f"Best Similarity: {best_results['target_similarity']:.2f}_")
        print(f"Random Walk pt and wav files ---> {results_dir}")

        return

    def cma_es_optimization(self, step_limit: int, sigma: float = 0.1, use_advanced_scoring: bool = True):
        """
        CMA-ES optimization for voice cloning.
        
        Args:
            step_limit: Maximum number of iterations
            sigma: Initial step size (0.1 is usually safe for normalized vectors)
            use_advanced_scoring: Use WavLM/Whisper scoring instead of Resemblyzer
        """
        if not CMA_AVAILABLE:
            raise RuntimeError("CMA-ES not available. Install with: pip install cma")
        
        print(f"Starting CMA-ES Optimization (Advanced Scoring: {use_advanced_scoring})")
        
        # Create Results Directory
        now = datetime.datetime.now()
        results_dir = Path(OUT_DIR / f'{self.output_name}_{self.target_audio.stem}_cma_{now.strftime("%Y%m%d_%H%M%S")}')
        os.makedirs(results_dir, exist_ok=True)
        
        # Setup optimizer
        initial_params = self.starting_voice.cpu().numpy()
        es = cma.CMAEvolutionStrategy(initial_params, sigma)
        
        # Track best solution
        best_loss = float('inf')
        best_voice = None
        iteration = 0
        
        # Track whether we're using advanced scoring (for fallback handling)
        using_advanced_scoring = use_advanced_scoring
        
        print(f"Initial vector shape: {initial_params.shape}")
        print(f"Sigma: {sigma}")
        
        # Optimization Loop
        while not es.stop() and iteration < step_limit:
            # ASK: Get candidate solutions
            solutions = es.ask()
            
            fitness_values = []
            
            for vector_candidate in solutions:
                # Convert to tensor
                style_vector = torch.tensor(vector_candidate).float()
                
                # Generate short audio sample for evaluation (faster)
                # Use a shorter text for speed during optimization
                short_text = self.target_text[:CMA_SHORT_TEXT_LENGTH] if len(self.target_text) > CMA_SHORT_TEXT_LENGTH else self.target_text
                audio = self.speech_generator.generate_audio(short_text, style_vector)
                
                # Calculate loss
                if using_advanced_scoring:
                    try:
                        loss = self.fitness_scorer.get_complex_score(audio, short_text)
                    except Exception as e:
                        print(f"Error in advanced scoring: {e}")
                        print("Falling back to standard scoring for this run")
                        using_advanced_scoring = False
                        # Fall back to standard scoring
                        results = self.score_voice(style_vector)
                        loss = -results["score"]  # Negate for minimization
                else:
                    # Use standard scoring (negated for minimization)
                    results = self.score_voice(style_vector)
                    loss = -results["score"]
                
                fitness_values.append(loss)
            
            # TELL: Report results to CMA-ES
            es.tell(solutions, fitness_values)
            
            # Track best solution
            current_best_idx = np.argmin(fitness_values)
            current_best_loss = fitness_values[current_best_idx]
            
            if current_best_loss < best_loss:
                best_loss = current_best_loss
                best_voice = torch.tensor(solutions[current_best_idx]).float()
                
                # Generate full audio for the best voice
                full_audio = self.speech_generator.generate_audio(self.target_text, best_voice)
                
                # Get detailed scores for logging
                if using_advanced_scoring:
                    # For advanced scoring, also compute traditional scores for comparison
                    trad_results = self.score_voice(best_voice)
                    print(f'Iteration:{iteration:<4} Loss:{-best_loss:.4f} (CMA) | '
                          f'Trad Score:{trad_results["score"]:.2f} | '
                          f'Target Sim:{trad_results["target_similarity"]:.3f}')
                else:
                    trad_results = self.score_voice(best_voice)
                    print(f'Iteration:{iteration:<4} Score:{trad_results["score"]:.2f} | '
                          f'Target Sim:{trad_results["target_similarity"]:.3f} | '
                          f'Self Sim:{trad_results["self_similarity"]:.3f}')
                
                # Save best voice
                torch.save(best_voice,
                          f'{results_dir}/{self.output_name}_{iteration}_{-best_loss:.4f}.pt')
                sf.write(
                    f'{results_dir}/{self.output_name}_{iteration}_{-best_loss:.4f}.wav',
                    full_audio, 24000)
            
            # Periodic display
            if iteration % 10 == 0:
                es.disp()
                clear_gpu_memory()
            
            iteration += 1
        
        # Print final results
        print("\nCMA-ES Optimization Complete")
        print(f"Total iterations: {iteration}")
        print(f"Best loss: {best_loss:.4f}")
        print(f"Results saved to: {results_dir}")
        
        # Save final best voice
        if best_voice is not None:
            torch.save(best_voice, f'{results_dir}/best_voice_final.pt')
            final_audio = self.speech_generator.generate_audio(self.target_text, best_voice)
            sf.write(f'{results_dir}/best_voice_final.wav', final_audio, 24000)
        
        return best_voice

    def score_voice(self,voice: torch.Tensor,min_similarity: float = 0.0) -> dict[str,Any]:
        """Using a harmonic mean calculation to provide a score for the voice in similarity"""
        audio = self.speech_generator.generate_audio(self.target_text, voice)
        target_similarity = self.fitness_scorer.target_similarity(audio)
        results: dict[str,Any] = {
            'audio': audio
        }
        # Bail early and save the compute if the similarity sucks
        if target_similarity > min_similarity:
            audio2 = self.speech_generator.generate_audio(self.other_text, voice)
            results.update(self.fitness_scorer.hybrid_similarity(audio,audio2,target_similarity))
        else:
            results["score"] = 0.0
            results["target_similarity"] = target_similarity

        return results
