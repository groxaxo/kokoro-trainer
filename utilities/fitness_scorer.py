# from scipy.spatial.distance import cosine
from typing import Any

import librosa
import numpy as np
import scipy.stats
import soundfile as sf
import torch
from numpy._typing import NDArray
from resemblyzer import preprocess_wav, VoiceEncoder

# Import models for advanced scoring
try:
    from transformers import Wav2Vec2FeatureExtractor, WavLMForXVector, WhisperProcessor, WhisperForConditionalGeneration
    import jiwer
    import torchaudio
    ADVANCED_MODELS_AVAILABLE = True
except ImportError:
    ADVANCED_MODELS_AVAILABLE = False

# Import Spanish language utilities
try:
    from utilities.spanish_utils import SpanishVoiceScorer
    SPANISH_UTILS_AVAILABLE = True
except ImportError:
    SPANISH_UTILS_AVAILABLE = False


class FitnessScorer:
    # Class-level shared encoder to avoid reinitializing for each instance
    _shared_encoder = None
    _shared_wavlm_extractor = None
    _shared_wavlm_model = None
    _shared_whisper_processor = None
    _shared_whisper_model = None
    
    @classmethod
    def get_encoder(cls):
        """Get or create shared voice encoder instance"""
        if cls._shared_encoder is None:
            cls._shared_encoder = VoiceEncoder()
        return cls._shared_encoder
    
    @classmethod
    def get_wavlm_models(cls):
        """Get or create shared WavLM models for speaker verification"""
        if not ADVANCED_MODELS_AVAILABLE:
            return None, None
        
        if cls._shared_wavlm_extractor is None:
            cls._shared_wavlm_extractor = Wav2Vec2FeatureExtractor.from_pretrained('microsoft/wavlm-base-plus-sv')
            cls._shared_wavlm_model = WavLMForXVector.from_pretrained('microsoft/wavlm-base-plus-sv')
        return cls._shared_wavlm_extractor, cls._shared_wavlm_model
    
    @classmethod
    def get_whisper_models(cls):
        """Get or create shared Whisper models for speech recognition"""
        if not ADVANCED_MODELS_AVAILABLE:
            return None, None
        
        if cls._shared_whisper_processor is None:
            cls._shared_whisper_processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")
            cls._shared_whisper_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny")
        return cls._shared_whisper_processor, cls._shared_whisper_model
    
    def __init__(self,target_path: str):
        self.encoder = self.get_encoder()
        self.target_audio, _ = sf.read(target_path,dtype="float32")
        self.target_wav = preprocess_wav(target_path,source_sr=24000)
        self.target_embed = self.encoder.embed_utterance(self.target_wav)
        self.target_features = self.extract_features(self.target_audio)
        
        # Initialize advanced models if available
        self.wavlm_extractor, self.wavlm_model = self.get_wavlm_models()
        self.whisper_processor, self.whisper_model = self.get_whisper_models()
        
        # Compute target WavLM embedding if available
        if ADVANCED_MODELS_AVAILABLE and self.wavlm_extractor is not None and self.wavlm_model is not None:
            # Resample to 16kHz for WavLM
            target_16k = torchaudio.functional.resample(
                torch.tensor(self.target_audio).unsqueeze(0), 
                orig_freq=24000, 
                new_freq=16000
            ).squeeze(0).numpy()
            
            inputs = self.wavlm_extractor(target_16k, return_tensors="pt", sampling_rate=16000)
            with torch.no_grad():
                self.target_wavlm_embed = self.wavlm_model(**inputs).embeddings
        else:
            self.target_wavlm_embed = None
        
        self.target_text = None  # Will be set if using advanced scoring
        
        # Initialize Spanish voice scorer if available
        self.spanish_scorer = SpanishVoiceScorer() if SPANISH_UTILS_AVAILABLE else None
        self.is_spanish_mode = False  # Will be set when Spanish text is detected

    def hybrid_similarity(self, audio: NDArray[np.float32], audio2: NDArray[np.float32],target_similarity: float):
        features = self.extract_features(audio)
        self_similarity = self.self_similarity(audio,audio2)
        target_features_pentalty = self.target_feature_penalty(features)

        #Normalize and make higher = better
        feature_similarity = (100.0 - target_features_pentalty) / 100.0
        if feature_similarity < 0.0:
            feature_similarity = 0.01

        values = [target_similarity, self_similarity, feature_similarity]
        # Playing around with the weights can greatly affect scoring and random walk behavior
        weights = [0.48,0.5,0.02]
        score = (np.sum(weights) / np.sum(np.array(weights) / np.array(values))) * 100.0
        
        # Apply Spanish language bonus if applicable
        if self.is_spanish_mode and self.spanish_scorer and self.target_text:
            spanish_bonus = self.spanish_scorer.get_spanish_quality_bonus(self.target_text, features)
            # Add bonus as a percentage increase (up to 5% improvement)
            score = score * (1.0 + spanish_bonus * 0.05)

        return {
            "score": score,
            "target_similarity": target_similarity,
            "self_similarity": self_similarity,
            "feature_similarity": feature_similarity
        }

    def get_complex_score(self, generated_audio: NDArray[np.float32], target_text: str) -> float:
        """
        Calculate a composite score using WavLM, Whisper, and quality metrics.
        This is the advanced scoring function for CMA-ES optimization.
        
        Args:
            generated_audio: The audio array from Kokoro (24kHz mono)
            target_text: The expected transcription text
            
        Returns:
            float: Negative score (loss) for minimization
        """
        if not ADVANCED_MODELS_AVAILABLE:
            raise RuntimeError("Advanced models not available. Install: transformers, jiwer, torchaudio")
        
        # Resample to 16kHz for models
        audio_16k = torchaudio.functional.resample(
            torch.tensor(generated_audio).unsqueeze(0), 
            orig_freq=24000, 
            new_freq=16000
        ).squeeze(0).numpy()
        
        # --- Metric 1: WavLM Similarity (Identity) ---
        inputs = self.wavlm_extractor(audio_16k, return_tensors="pt", sampling_rate=16000)
        with torch.no_grad():
            gen_embedding = self.wavlm_model(**inputs).embeddings
        
        # Cosine Similarity (Higher is better)
        sim_score = torch.nn.functional.cosine_similarity(
            gen_embedding, self.target_wavlm_embed
        ).item()
        
        # --- Metric 2: Whisper WER (Intelligibility) ---
        input_features = self.whisper_processor(
            audio_16k, sampling_rate=16000, return_tensors="pt"
        ).input_features
        predicted_ids = self.whisper_model.generate(input_features)
        transcription = self.whisper_processor.batch_decode(
            predicted_ids, skip_special_tokens=True
        )[0]
        
        # Calculate Word Error Rate (Lower is better)
        wer_score = jiwer.wer(target_text, transcription)
        
        # --- Metric 3: Quality (Placeholder) ---
        # For now, use a simple quality proxy based on audio characteristics
        # In a real implementation, you would use NISQA or DNSMOS
        quality_mos = self._predict_quality_proxy(generated_audio)
        normalized_quality = quality_mos / 5.0  # Scale to 0-1
        
        # --- Final Equation ---
        # Score = (Sim * 0.5) + (Quality * 0.3) - (WER * 0.2)
        # Return negative because CMA minimizes
        final_score = (sim_score * 0.5) + (normalized_quality * 0.3) - (wer_score * 0.2)
        
        return -final_score  # Negative for minimization
    
    def _predict_quality_proxy(self, audio: NDArray[np.float32]) -> float:
        """
        Simple quality proxy based on audio features.
        In production, replace with NISQA or DNSMOS.
        
        Returns MOS score (1-5, higher is better)
        """
        # Use feature-based quality estimation
        features = self.extract_features(audio)
        
        # Simple heuristic: penalize extreme values
        quality = 5.0
        
        # Penalize very high zero crossing rate (noise)
        if features["zero_crossing_rate"] > 0.2:
            quality -= 1.0
        
        # Penalize very low RMS energy (too quiet)
        if features["rms_energy"] < 0.01:
            quality -= 0.5
        
        # Penalize very high spectral flatness (noise)
        if features["spectral_flatness_mean"] > 0.8:
            quality -= 1.0
        
        return max(1.0, min(5.0, quality))

    def target_similarity(self,audio: NDArray[np.float32]) -> float:
        audio_wav = preprocess_wav(audio,source_sr=24000)
        audio_embed = self.encoder.embed_utterance(audio_wav)
        similarity = np.inner(audio_embed, self.target_embed)
        return similarity

    def target_feature_penalty(self,features: dict[str, Any]) -> float:
        """Penalizes for differences in audio features"""
        # Normalized feature difference compared to target features
        penalty = 0.0
        for key, value in features.items():
            target_value = self.target_features[key]
            # Prevent division by zero
            if target_value != 0:
                diff = abs((value - target_value) / target_value)
            else:
                diff = abs(value)
            penalty += diff
        return penalty

    def self_similarity(self,audio1: NDArray[np.float32], audio2: NDArray[np.float32]) -> float:
        """Self similarity indicates model stability. Poor self similarity means different input makes different sounding voices"""
        audio_wav1 = preprocess_wav(audio1,source_sr=24000)
        audio_embed1 = self.encoder.embed_utterance(audio_wav1)

        audio_wav2 = preprocess_wav(audio2,source_sr=24000)
        audio_embed2 = self.encoder.embed_utterance(audio_wav2)
        return np.inner(audio_embed1, audio_embed2)

    def extract_features(self, audio: NDArray[np.float32] | NDArray[np.float64], sr: int = 24000) -> dict[str, Any]:
        """
        Extract a comprehensive set of audio features for fingerprinting speech segments.

        Args:
            audio: Audio signal as numpy array (np.float32)
            sr: Sample rate (fixed at 24000 Hz)

        Returns:
            Dictionary containing extracted features
        """
        # Ensure audio is the right shape (flatten stereo to mono if needed)
        if len(audio.shape) > 1 and audio.shape[1] > 1:
            audio = np.mean(audio, axis=1)

        # Initialize features dictionary
        features = {}

        # Basic features
        # features["duration"] = len(audio) / sr
        features["rms_energy"] = float(np.sqrt(np.mean(audio**2)))
        features["zero_crossing_rate"] = float(np.mean(librosa.feature.zero_crossing_rate(audio)))

        # Spectral features
        # Compute STFT
        n_fft = 2048  # Window size
        hop_length = 512  # Hop length

        # Spectral centroid and bandwidth (where the "center" of the sound is)
        spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length)[0]
        features["spectral_centroid_mean"] = float(np.mean(spectral_centroids))
        features["spectral_centroid_std"] = float(np.std(spectral_centroids))

        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length)[0]
        features["spectral_bandwidth_mean"] = float(np.mean(spectral_bandwidth))
        features["spectral_bandwidth_std"] = float(np.std(spectral_bandwidth))

        # Spectral rolloff
        rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length)[0]
        features["spectral_rolloff_mean"] = float(np.mean(rolloff))
        features["spectral_rolloff_std"] = float(np.std(rolloff))

        # Spectral contrast
        contrast = librosa.feature.spectral_contrast(y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length)
        features["spectral_contrast_mean"] = float(np.mean(contrast))
        features["spectral_contrast_std"] = float(np.std(contrast))

        # MFCCs (Mel-frequency cepstral coefficients) - important for speech
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13, n_fft=n_fft, hop_length=hop_length)

        # Store each MFCC coefficient mean and std
        for i in range(len(mfccs)):
            features[f"mfcc{i+1}_mean"] = float(np.mean(mfccs[i]))
            features[f"mfcc{i+1}_std"] = float(np.std(mfccs[i]))

        # MFCC delta features (first derivative)
        mfcc_delta = librosa.feature.delta(mfccs)
        for i in range(len(mfcc_delta)):
            features[f"mfcc{i+1}_delta_mean"] = float(np.mean(mfcc_delta[i]))
            features[f"mfcc{i+1}_delta_std"] = float(np.std(mfcc_delta[i]))

        # Chroma features - useful for characterizing harmonic content
        chroma = librosa.feature.chroma_stft(y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length)
        features["chroma_mean"] = float(np.mean(chroma))
        features["chroma_std"] = float(np.std(chroma))

        # Store individual chroma features
        for i in range(len(chroma)):
            features[f"chroma_{i+1}_mean"] = float(np.mean(chroma[i]))
            features[f"chroma_{i+1}_std"] = float(np.std(chroma[i]))

        # Mel spectrogram (average across frequency bands)
        mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length)
        features["mel_spec_mean"] = float(np.mean(mel_spec))
        features["mel_spec_std"] = float(np.std(mel_spec))

        # Spectral flatness - measure of the noisiness of the signal
        flatness = librosa.feature.spectral_flatness(y=audio, n_fft=n_fft, hop_length=hop_length)[0]
        features["spectral_flatness_mean"] = float(np.mean(flatness))
        features["spectral_flatness_std"] = float(np.std(flatness))

        # Tonnetz (tonal centroid features)
        tonnetz = librosa.feature.tonnetz(y=audio, sr=sr)
        features["tonnetz_mean"] = float(np.mean(tonnetz))
        features["tonnetz_std"] = float(np.std(tonnetz))

        # Rhythm features - tempo and beat strength
        tempo, beat_frames = librosa.beat.beat_track(y=audio, sr=sr)
        features["tempo"] = float(tempo)

        if len(beat_frames) > 0:
            # Calculate beat_stats only if beats are detected
            beat_times = librosa.frames_to_time(beat_frames, sr=sr)
            if len(beat_times) > 1:
                beat_diffs = np.diff(beat_times)
                features["beat_mean"] = float(np.mean(beat_diffs))
                features["beat_std"] = float(np.std(beat_diffs))
            else:
                features["beat_mean"] = 0.0
                features["beat_std"] = 0.0
        else:
            features["beat_mean"] = 0.0
            features["beat_std"] = 0.0

        # Pitch and harmonics
        pitches, magnitudes = librosa.core.piptrack(y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length)

        # For each frame, find the highest magnitude pitch
        pitch_values = []
        for i in range(magnitudes.shape[1]):
            index = magnitudes[:, i].argmax()
            pitch = pitches[index, i]
            if pitch > 0:  # Exclude zero pitch
                pitch_values.append(pitch)

        if pitch_values:
            features["pitch_mean"] = float(np.mean(pitch_values))
            features["pitch_std"] = float(np.std(pitch_values))
        else:
            features["pitch_mean"] = 0.0
            features["pitch_std"] = 0.0

        # Speech-specific features

        # Voice Activity Detection (simplified)
        # Higher energies typically indicate voice activity
        energy = np.array([sum(abs(audio[i:i+hop_length])) for i in range(0, len(audio), hop_length)])
        features["energy_mean"] = float(np.mean(energy))
        features["energy_std"] = float(np.std(energy))

        # Harmonics-to-noise ratio (simplified approximation)
        # Using the squared magnitude of the spectrogram
        S = np.abs(librosa.stft(audio, n_fft=n_fft, hop_length=hop_length))
        S_squared = S**2
        S_mean = np.mean(S_squared, axis=1)
        S_std = np.std(S_squared, axis=1)
        S_ratio = np.divide(S_mean, S_std, out=np.zeros_like(S_mean), where=S_std!=0)
        features["harmonic_ratio"] = float(np.mean(S_ratio))

        # Statistical features from the raw waveform
        features["audio_mean"] = float(np.mean(audio))
        features["audio_std"] = float(np.std(audio))
        features["audio_skew"] = float(scipy.stats.skew(audio))
        features["audio_kurtosis"] = float(scipy.stats.kurtosis(audio))

        return features
