# Latin American Spanish Voice Training Guide

## 🇪🇸 Overview

This guide provides specific instructions and best practices for training high-quality Latin American Spanish voices using KVoiceWalk. The system now includes specialized features that automatically detect and optimize for Spanish language characteristics.

## ✨ Spanish-Specific Features

### Automatic Language Detection
KVoiceWalk automatically detects when you're working with Spanish text and enables Spanish-optimized training:

```bash
uv run main.py \
  --target_audio ./mi_voz.wav \
  --target_text "El español es un idioma hermoso con características únicas" \
  --use_cma_es \
  --use_super_seed
```

When Spanish is detected, you'll see:
```
🇪🇸 Spanish language detected (confidence: 87%)
   Enabling Spanish-optimized training parameters...
```

### Spanish Language Enhancements

1. **Phonetic Feature Analysis**: Considers Spanish-specific phonemes and pronunciation patterns
2. **Prosody Scoring**: Evaluates syllable-timed rhythm (characteristic of Spanish)
3. **Accent Preservation**: Maintains Spanish diacritics (á, é, í, ó, ú, ñ, ü)
4. **Quality Bonuses**: Rewards clear vowel pronunciation and appropriate rhythm

## 📋 Best Practices for Latin American Spanish

### 1. Text Selection

**Ideal Characteristics:**
- **Length**: 25-40 words (Spanish sentences tend to be longer than English)
- **Content**: Natural, conversational Spanish
- **Punctuation**: Include inverted marks (¿...? and ¡...!)
- **Vocabulary**: Use common, everyday words

**Good Example:**
```
¿Cómo está el clima hoy en Buenos Aires? Hace mucho calor y necesito 
encontrar un lugar con aire acondicionado para trabajar tranquilamente.
```

**What to Avoid:**
- Machine-translated text (often unnatural)
- Very short phrases (< 15 words)
- Technical jargon without context
- Mixed Spanish-English code-switching (unless that's your target)

### 2. Audio Quality Requirements

**Target Audio Specifications:**
- **Format**: WAV, mono, 24kHz (auto-converted if different)
- **Duration**: 20-30 seconds ideal
- **Quality**: Clear pronunciation, minimal background noise
- **Speaker**: Single speaker, native or near-native accent
- **Characteristics**: 
  - Clear vowel sounds (critical for Spanish)
  - Consistent rhythm (syllable-timed)
  - Natural intonation patterns

### 3. Regional Variations

Latin American Spanish has several regional variations. Consider your target dialect:

#### Mexican Spanish
- Softer pronunciation
- Distinctive intonation patterns
- Conservative consonant pronunciation

**Example Text:**
```
Buenos días, ¿cómo estás? Voy al mercado a comprar algunas verduras 
para preparar la comida de hoy.
```

#### Argentinian Spanish
- Yeísmo (ll/y → sh sound)
- Voseo (use of "vos" instead of "tú")
- Distinctive rhythm

**Example Text:**
```
¿Vos sabés dónde queda la biblioteca? Necesito encontrar unos libros 
para mi proyecto de la universidad.
```

#### Colombian Spanish
- Clear, neutral pronunciation
- Often considered "neutral" for dubbing
- Slower pace

**Example Text:**
```
Me encanta este lugar porque es muy tranquilo y puedo concentrarme 
en mi trabajo sin distracciones.
```

#### Caribbean Spanish (Cuba, Puerto Rico, Dominican Republic)
- Faster pace
- Consonant weakening
- Aspirated 's' sounds

**Example Text:**
```
¿Qué tú quieres hacer hoy? Podemos ir a la playa o quedarnos 
en casa viendo películas.
```

## 🚀 Recommended Training Parameters

### For Latin American Spanish

```bash
uv run main.py \
  --target_audio ./spanish_voice.wav \
  --target_text "Tu texto en español aquí" \
  --use_cma_es \
  --use_super_seed \
  --step_limit 500 \
  --population_limit 12 \
  --cma_sigma 0.12 \
  --lang_code a
```

**Parameter Explanations:**

- `--step_limit 500`: Spanish may need slightly more iterations for quality
- `--population_limit 12`: Higher diversity helps capture Spanish characteristics
- `--cma_sigma 0.12`: Slightly higher exploration for accent variation
- `--lang_code a`: Auto-detect (or use 'es' if specifically targeting Spanish)

### Quick Start Example

```bash
# Step 1: Transcribe your Spanish audio
uv run main.py \
  --target_audio ./mi_voz_español.wav \
  --transcribe_start

# Step 2: Train with Spanish optimizations
uv run main.py \
  --target_audio ./mi_voz_español.wav \
  --target_text ./texts/mi_voz_español.txt \
  --use_cma_es \
  --use_super_seed \
  --step_limit 400 \
  --output_name mi_voz_latina

# Step 3: Test the result
uv run main.py \
  --test_voice ./out/mi_voz_latina_*/best_voice_final.pt \
  --target_text "¡Hola! Esta es una prueba de mi voz clonada en español."
```

## 🎯 Optimization Tips

### 1. Text Preparation

**Normalize Your Text:**
```python
from utilities.spanish_utils import SpanishTextNormalizer

normalizer = SpanishTextNormalizer()
normalized_text = normalizer.normalize(
    "Dr. García fue a EE.UU. el lunes",
    preserve_accents=True
)
# Result: "Doctor García fue a Estados Unidos el lunes"
```

**Validate Text Quality:**
```python
from utilities.spanish_utils import SpanishVoiceScorer

scorer = SpanishVoiceScorer()
recommendations = scorer.get_latin_american_recommendations(your_text)
print(recommendations['quality_tips'])
```

### 2. Training Strategy

**Progressive Approach:**

1. **Initial Test (50-100 steps):**
   ```bash
   uv run main.py --target_audio audio.wav --target_text "texto" --step_limit 100
   ```
   Quick validation that everything works.

2. **Standard Training (300-500 steps):**
   ```bash
   uv run main.py --target_audio audio.wav --target_text "texto" \
     --use_cma_es --step_limit 400
   ```
   Good quality for most use cases.

3. **High-Quality Training (500-1000 steps):**
   ```bash
   uv run main.py --target_audio audio.wav --target_text "texto" \
     --use_cma_es --use_super_seed --use_advanced_scoring --step_limit 800
   ```
   Maximum quality for production use.

### 3. Multi-Dialect Training

If you need a voice that works across multiple Latin American dialects:

1. Use **neutral Colombian or Mexican** audio as base
2. Increase population diversity: `--population_limit 15`
3. Use longer training: `--step_limit 600`
4. Test with text from different regions

## 📊 Quality Evaluation

### What to Listen For

**Good Spanish Voice:**
- ✅ Clear vowel sounds (a, e, i, o, u)
- ✅ Proper syllable-timed rhythm
- ✅ Natural intonation (rising for questions, etc.)
- ✅ Correct pronunciation of ñ, rr, ll
- ✅ Appropriate speed (not too fast or slow)

**Warning Signs:**
- ❌ Muddy vowels
- ❌ Stress-timed rhythm (sounds like English)
- ❌ Monotone delivery
- ❌ Incorrect consonant sounds
- ❌ Unnatural pauses

### Spanish-Specific Metrics

The system automatically calculates:

1. **Vowel Clarity**: RMS energy analysis
2. **Syllable Consistency**: Rhythm ratio (higher is better)
3. **Spanish Confidence**: Language detection score
4. **Prosodic Quality**: Intonation patterns

These are combined into a **Spanish Quality Bonus** (up to 5% improvement on overall score).

## 🔧 Troubleshooting

### Issue: "Text may not be Spanish"

**Solution:**
- Ensure you're using actual Spanish text
- Include Spanish-specific characters (á, é, í, ó, ú, ñ)
- Use common Spanish words

### Issue: Poor quality despite good settings

**Possible causes:**
1. **Target audio quality**: Check if source audio is clear
2. **Text mismatch**: Ensure transcription matches audio exactly
3. **Regional mismatch**: Audio and text from different dialects
4. **Duration**: Audio too short (< 15 seconds) or too long (> 40 seconds)

**Solutions:**
```bash
# Re-transcribe to ensure accuracy
uv run main.py --target_audio audio.wav --transcribe_start

# Use longer training
--step_limit 600

# Try super-seed initialization
--use_super_seed
```

### Issue: Accent sounds wrong

**Solution:**
- Verify your target audio has the desired accent
- Increase population limit: `--population_limit 15`
- Try using `--interpolate_start` for better initialization
- Consider training with multiple audio samples from same speaker

## 📚 Example Workflows

### Workflow 1: Mexican Spanish News Reader

```bash
# Good for: Clear, professional narration
uv run main.py \
  --target_audio mexican_news.wav \
  --target_text "Las noticias de hoy incluyen importantes desarrollos..." \
  --use_cma_es \
  --use_super_seed \
  --step_limit 500 \
  --output_name locutor_mexicano
```

### Workflow 2: Argentinian Conversational

```bash
# Good for: Natural dialogue, casual speech
uv run main.py \
  --target_audio argentino_conversacion.wav \
  --target_text "Che, ¿vos sabés qué hora es? Tengo que..." \
  --use_cma_es \
  --step_limit 400 \
  --cma_sigma 0.15 \
  --output_name voz_argentina
```

### Workflow 3: Colombian Neutral (Dubbing Quality)

```bash
# Good for: Professional dubbing, audiobooks
uv run main.py \
  --target_audio colombiano_neutral.wav \
  --target_text "En este capítulo, nuestro protagonista descubre..." \
  --use_cma_es \
  --use_super_seed \
  --use_advanced_scoring \
  --step_limit 600 \
  --output_name doblaje_neutral
```

## 🌟 Advanced Features

### Custom Spanish Scoring

The system uses enhanced scoring for Spanish:

```python
# In fitness_scorer.py
if is_spanish_mode:
    spanish_bonus = scorer.get_spanish_quality_bonus(text, features)
    score = score * (1.0 + spanish_bonus * 0.05)
```

This rewards:
- High Spanish language confidence (30%)
- Appropriate word length distribution (20%)
- Syllable-timed rhythm (30%)
- Clear vowel pronunciation (20%)

### Language-Specific Text Normalization

Automatic expansion of Spanish abbreviations:
- Dr. → Doctor
- Sra. → Señora
- EE.UU. → Estados Unidos
- Ud. → Usted

## 📖 Additional Resources

### Spanish Phonetics Reference

**Key Sounds:**
- **Vowels**: /a/, /e/, /i/, /o/, /u/ (pure, monophthong)
- **Diphthongs**: /ai/, /ei/, /oi/, /au/, /eu/
- **Consonants**: 
  - /ñ/ (niño)
  - /rr/ (perro) 
  - /ll/ → /y/ in Latin America (yeísmo)
  - /z/, /c/ → /s/ in Latin America (seseo)

### Recommended Text Length by Type

| Type | Words | Characters | Example |
|------|-------|------------|---------|
| **Short** | 15-25 | 100-150 | Quick tests, greetings |
| **Standard** | 25-40 | 150-250 | Most training, general use |
| **Long** | 40-60 | 250-400 | Audiobooks, narration |

### Quality Checklist

Before starting training:
- [ ] Audio is clear, 20-30 seconds
- [ ] Text is natural Spanish
- [ ] Text matches audio exactly
- [ ] Regional accent is consistent
- [ ] Background noise is minimal
- [ ] Speaker is native or near-native
- [ ] Punctuation includes ¿? and ¡!

## 🤝 Contributing

Found ways to improve Spanish voice training? Please share:
1. Regional-specific optimizations
2. Better text examples
3. Quality improvement tips
4. Dialect-specific parameters

## ⚠️ Important Notes

1. **Accent Preservation**: Always use `preserve_accents=True` (default) for Spanish
2. **Whisper Transcription**: May not always preserve Spanish punctuation (¿!)
3. **Model Limitations**: Based on Kokoro model capabilities
4. **Training Time**: Spanish may take 10-20% longer than English for same quality

## 📞 Support

For Spanish-specific issues:
1. Check text is valid Spanish: `SpanishTextNormalizer().validate_for_tts(text)`
2. Verify language detection: Look for 🇪🇸 in output
3. Review quality tips in recommendations
4. Try increasing `--step_limit` and `--population_limit`

---

**¡Buena suerte con tu entrenamiento de voz en español! 🎤**

*Good luck with your Spanish voice training!*
