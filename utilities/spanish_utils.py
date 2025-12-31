"""
Spanish Language Utilities for Voice Training

This module provides specialized tools for training and evaluating Spanish voices,
with particular focus on Latin American Spanish characteristics.
"""

import re
from typing import Dict, List, Tuple, Any
import numpy as np


# Spanish phoneme groups for Latin American Spanish
SPANISH_VOWELS = ['a', 'e', 'i', 'o', 'u']
SPANISH_DIPHTHONGS = ['ai', 'au', 'ei', 'eu', 'oi', 'ou', 'ia', 'ie', 'io', 'iu', 'ua', 'ue', 'ui', 'uo']

# Consonants that vary across Latin American regions
LATIN_AMERICAN_CONSONANTS = {
    'll': 'y',  # Yeísmo (common in Latin America)
    'z': 's',   # Seseo (common in Latin America)
    'c': 's',   # Seseo (before e, i)
}

# Common Spanish word patterns for validation
SPANISH_PATTERNS = {
    'articles': ['el', 'la', 'los', 'las', 'un', 'una', 'unos', 'unas'],
    'pronouns': ['yo', 'tú', 'él', 'ella', 'nosotros', 'nosotras', 'vosotros', 'vosotras', 'ellos', 'ellas'],
    'common_words': ['es', 'no', 'sí', 'de', 'que', 'en', 'por', 'para', 'con', 'su', 'mi', 'tu']
}

# Detection thresholds and weights
SPANISH_CHAR_WEIGHT = 20  # Multiplier for Spanish character ratio
SPANISH_CHAR_MAX_SCORE = 0.5  # Maximum score contribution from Spanish characters
SPANISH_WORD_WEIGHT = 2  # Multiplier for Spanish word ratio
SPANISH_WORD_MAX_SCORE = 0.5  # Maximum score contribution from Spanish words
SPANISH_DETECTION_THRESHOLD_WITH_CHARS = 0.2  # Lower threshold when Spanish chars present
SPANISH_DETECTION_THRESHOLD_DEFAULT = 0.3  # Default threshold


class SpanishTextNormalizer:
    """Normalize and validate Spanish text for TTS processing"""
    
    def __init__(self):
        self.accent_map = {
            'á': 'a', 'é': 'e', 'í': 'i', 'ó': 'o', 'ú': 'u',
            'Á': 'A', 'É': 'E', 'Í': 'I', 'Ó': 'O', 'Ú': 'U',
            'ñ': 'n', 'Ñ': 'N', 'ü': 'u', 'Ü': 'U'
        }
    
    def is_spanish_text(self, text: str) -> Tuple[bool, float]:
        """
        Detect if text is likely Spanish
        
        Returns:
            Tuple of (is_spanish, confidence_score)
        """
        if not text or len(text.strip()) == 0:
            return False, 0.0
        
        text_lower = text.lower()
        words = text_lower.split()
        
        if len(words) == 0:
            return False, 0.0
        
        # Check for Spanish-specific characters (strong indicator)
        spanish_chars = sum(1 for c in text if c in 'áéíóúñü¿¡')
        
        # Check for common Spanish words
        spanish_word_count = 0
        for word_list in SPANISH_PATTERNS.values():
            spanish_word_count += sum(1 for word in words if word in word_list)
        
        # Calculate confidence with improved weights
        char_score = min(spanish_chars / max(len(text), 1) * SPANISH_CHAR_WEIGHT, SPANISH_CHAR_MAX_SCORE)
        word_score = min(spanish_word_count / max(len(words), 1) * SPANISH_WORD_WEIGHT, SPANISH_WORD_MAX_SCORE)
        
        confidence = char_score + word_score
        
        # Lower threshold for short text if Spanish characters present
        threshold = SPANISH_DETECTION_THRESHOLD_WITH_CHARS if spanish_chars > 0 else SPANISH_DETECTION_THRESHOLD_DEFAULT
        
        return confidence > threshold, confidence
    
    def normalize(self, text: str, preserve_accents: bool = True) -> str:
        """
        Normalize Spanish text for TTS
        
        Args:
            text: Input Spanish text
            preserve_accents: Keep accent marks (recommended for Spanish)
            
        Returns:
            Normalized text
        """
        # Basic normalization
        normalized = text.strip()
        
        # Normalize whitespace
        normalized = re.sub(r'\s+', ' ', normalized)
        
        # Expand common abbreviations
        normalized = self._expand_abbreviations(normalized)
        
        # Remove accents if requested (not recommended for Spanish)
        if not preserve_accents:
            for accented, plain in self.accent_map.items():
                normalized = normalized.replace(accented, plain)
        
        return normalized
    
    def _expand_abbreviations(self, text: str) -> str:
        """Expand common Spanish abbreviations"""
        abbreviations = {
            r'\bDr\.': 'Doctor',
            r'\bDra\.': 'Doctora',
            r'\bSr\.': 'Señor',
            r'\bSra\.': 'Señora',
            r'\bSrta\.': 'Señorita',
            r'\bEE\.UU\.': 'Estados Unidos',
            r'\bUd\.': 'Usted',
            r'\bUds\.': 'Ustedes',
        }
        
        result = text
        for abbr, expansion in abbreviations.items():
            result = re.sub(abbr, expansion, result, flags=re.IGNORECASE)
        
        return result
    
    def validate_for_tts(self, text: str) -> Dict[str, Any]:
        """
        Validate text for Spanish TTS
        
        Returns:
            Dictionary with validation results
        """
        is_spanish, confidence = self.is_spanish_text(text)
        
        # Check length (Spanish words are slightly longer on average than English)
        words = text.split()
        avg_word_length = np.mean([len(w) for w in words]) if words else 0
        
        # Spanish average word length is about 4.5-5.5 characters
        length_appropriate = 4.0 <= avg_word_length <= 7.0
        
        # Check for special characters
        has_inverted_punctuation = '¿' in text or '¡' in text
        
        return {
            'is_spanish': is_spanish,
            'confidence': confidence,
            'length_appropriate': length_appropriate,
            'avg_word_length': avg_word_length,
            'has_spanish_punctuation': has_inverted_punctuation,
            'word_count': len(words),
            'recommendations': self._get_recommendations(is_spanish, confidence, length_appropriate)
        }
    
    def _get_recommendations(self, is_spanish: bool, confidence: float, length_ok: bool) -> List[str]:
        """Generate recommendations for better Spanish TTS"""
        recommendations = []
        
        if not is_spanish or confidence < 0.5:
            recommendations.append("Text may not be Spanish. Consider using Spanish text for better results.")
        
        if not length_ok:
            recommendations.append("Consider using more natural Spanish sentence structures.")
        
        if confidence > 0.7:
            recommendations.append("Text appears to be good Spanish. Ready for training.")
        
        return recommendations


class SpanishProsodicFeatures:
    """
    Extract prosodic features specific to Latin American Spanish
    
    Spanish prosody characteristics:
    - Syllable-timed rhythm (unlike stress-timed English)
    - Different intonation patterns
    - Vowel clarity and length
    """
    
    @staticmethod
    def estimate_syllable_count(text: str) -> int:
        """
        Estimate syllable count in Spanish text
        Spanish has clear syllable boundaries
        """
        # Remove punctuation
        text = re.sub(r'[^\w\s]', '', text.lower())
        
        # Count vowel groups (approximate syllables in Spanish)
        vowel_pattern = r'[aeiouáéíóúü]+'
        syllables = len(re.findall(vowel_pattern, text))
        
        return syllables
    
    @staticmethod
    def get_stress_pattern(text: str) -> List[bool]:
        """
        Identify stress patterns in Spanish words
        
        Spanish stress rules:
        - Words ending in vowel, n, or s: stress on penultimate syllable
        - Words ending in consonant (except n, s): stress on last syllable
        - Accents override default rules
        """
        words = text.lower().split()
        stress_pattern = []
        
        for word in words:
            # Check if word has accent mark (explicit stress)
            has_accent = any(c in word for c in 'áéíóú')
            stress_pattern.append(has_accent)
        
        return stress_pattern
    
    @staticmethod
    def calculate_rhythm_ratio(text: str) -> float:
        """
        Calculate rhythm ratio for Spanish (syllable-timed language)
        
        Returns:
            Ratio indicating syllable-timing consistency (higher = more consistent)
        """
        words = text.split()
        if len(words) < 2:
            return 1.0
        
        syllable_counts = [SpanishProsodicFeatures.estimate_syllable_count(w) for w in words]
        
        if not syllable_counts:
            return 1.0
        
        # Calculate coefficient of variation (lower = more consistent syllable timing)
        mean_syllables = np.mean(syllable_counts)
        std_syllables = np.std(syllable_counts)
        
        if mean_syllables == 0:
            return 1.0
        
        cv = std_syllables / mean_syllables
        
        # Convert to ratio (0-1, higher is better)
        consistency_ratio = max(0, 1 - cv)
        
        return consistency_ratio


class SpanishVoiceScorer:
    """
    Enhanced scoring for Spanish voice quality
    """
    
    def __init__(self):
        self.normalizer = SpanishTextNormalizer()
        self.prosody = SpanishProsodicFeatures()
    
    def get_spanish_quality_bonus(self, text: str, audio_features: Dict) -> float:
        """
        Calculate bonus score for Spanish-specific quality
        
        Args:
            text: The Spanish text that was spoken
            audio_features: Audio features from FitnessScorer
            
        Returns:
            Bonus score (0.0 to 1.0) to add to overall score
        """
        # Validate Spanish text
        validation = self.normalizer.validate_for_tts(text)
        
        if not validation['is_spanish']:
            return 0.0
        
        bonus = 0.0
        
        # Bonus for high Spanish confidence
        bonus += validation['confidence'] * 0.3
        
        # Bonus for appropriate word length
        if validation['length_appropriate']:
            bonus += 0.2
        
        # Bonus for syllable-timed rhythm
        rhythm_ratio = self.prosody.calculate_rhythm_ratio(text)
        bonus += rhythm_ratio * 0.3
        
        # Bonus for clear vowels (check RMS energy is sufficient)
        if audio_features.get('rms_energy', 0) > 0.02:
            bonus += 0.2
        
        return min(bonus, 1.0)
    
    def get_latin_american_recommendations(self, text: str) -> Dict[str, Any]:
        """
        Get recommendations specific to Latin American Spanish training
        
        Returns:
            Dictionary with recommendations and tips
        """
        validation = self.normalizer.validate_for_tts(text)
        
        recommendations = {
            'language_detected': 'Spanish' if validation['is_spanish'] else 'Unknown',
            'confidence': validation['confidence'],
            'quality_tips': [],
            'training_params': {}
        }
        
        if validation['is_spanish'] and validation['confidence'] > 0.5:
            recommendations['quality_tips'].extend([
                "✓ Spanish text detected - good for Latin American training",
                "Ensure audio includes clear vowel pronunciation",
                "Latin American Spanish uses seseo (s/z/c sound)",
                "Consider regional variations (Mexican, Argentinian, etc.)"
            ])
            
            # Recommended training parameters for Spanish
            recommendations['training_params'] = {
                'step_limit': 500,  # May need more steps for Spanish
                'use_cma_es': True,
                'use_super_seed': True,
                'population_limit': 12,  # Slightly higher for dialect variation
                'cma_sigma': 0.12,  # Slightly higher exploration
            }
        else:
            recommendations['quality_tips'].append(
                "⚠ Text may not be Spanish - consider using Spanish text for better results"
            )
        
        # Check for ideal text length (Spanish sentences tend to be longer)
        word_count = validation['word_count']
        if word_count < 20:
            recommendations['quality_tips'].append(
                f"Text is short ({word_count} words). Consider 25-40 words for Spanish training."
            )
        elif word_count > 60:
            recommendations['quality_tips'].append(
                f"Text is long ({word_count} words). Consider splitting into shorter segments."
            )
        else:
            recommendations['quality_tips'].append(
                f"✓ Text length ({word_count} words) is appropriate for Spanish training."
            )
        
        return recommendations
