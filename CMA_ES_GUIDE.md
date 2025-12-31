# CMA-ES Optimization Guide for KVoiceWalk

## Overview

This guide covers the new CMA-ES (Covariance Matrix Adaptation Evolution Strategy) optimization features added to KVoiceWalk. These features provide professional-grade voice cloning with advanced evaluation metrics.

## New Features

### 1. **Super-Seed Initialization**
Instead of starting from a single closest voice, the super-seed approach averages the top 5 most similar voices for a more robust starting point.

**Benefits:**
- More stable convergence
- Better initial quality
- Reduced risk of local optima

### 2. **CMA-ES Optimization**
Replaces random walk with a professional optimization algorithm that learns the direction of improvement.

**Benefits:**
- Faster convergence to optimal solutions
- Intelligent exploration of the voice space
- Better final results with fewer iterations

### 3. **Advanced Scoring with WavLM and Whisper**
Multi-metric evaluation that considers:
- **Identity** (WavLM): Is it the right person speaking?
- **Intelligibility** (Whisper WER): Are the words clear?
- **Quality** (proxy): Is the audio clean?

**Benefits:**
- More accurate voice quality assessment
- Better intelligibility in final output
- Closer speaker identity matching

## Installation

Install the required dependencies:

```bash
pip install cma transformers torchaudio jiwer
```

Or if using `uv`:

```bash
uv sync
```

## Usage

### Basic CMA-ES Optimization

```bash
python main.py \
  --target_audio ./example/target.wav \
  --target_text "Your target transcription here" \
  --use_cma_es \
  --step_limit 1000
```

### With Super-Seed Initialization

```bash
python main.py \
  --target_audio ./example/target.wav \
  --target_text "Your target transcription here" \
  --use_cma_es \
  --use_super_seed \
  --step_limit 1000
```

### With Advanced Scoring (WavLM + Whisper)

```bash
python main.py \
  --target_audio ./example/target.wav \
  --target_text "Your target transcription here" \
  --use_cma_es \
  --use_super_seed \
  --use_advanced_scoring \
  --step_limit 500
```

**Note:** Advanced scoring is computationally intensive. Reduce `--step_limit` when using it.

### Custom CMA-ES Parameters

Adjust the initial step size (sigma):

```bash
python main.py \
  --target_audio ./example/target.wav \
  --target_text "Your target transcription here" \
  --use_cma_es \
  --cma_sigma 0.2 \
  --step_limit 1000
```

**Sigma Guidelines:**
- `0.05-0.1`: Conservative, slower but stable
- `0.1-0.2`: Default, balanced exploration
- `0.2-0.3`: Aggressive, faster but less stable

## Complete Example

Here's a complete workflow using all new features:

```bash
# Step 1: Transcribe your audio (if needed)
python main.py \
  --target_audio ./my_voice.wav \
  --transcribe_start

# Step 2: Run CMA-ES optimization with super-seed
python main.py \
  --target_audio ./my_voice.wav \
  --target_text ./texts/my_voice.txt \
  --use_cma_es \
  --use_super_seed \
  --use_advanced_scoring \
  --step_limit 300 \
  --output_name my_voice_clone

# Step 3: Test the best result
python main.py \
  --test_voice ./out/my_voice_clone_*/best_voice_final.pt \
  --target_text "Hello, this is a test of my cloned voice!"
```

## Comparison: Random Walk vs CMA-ES

| Feature | Random Walk | CMA-ES |
|---------|-------------|--------|
| **Strategy** | Random mutations | Intelligent evolution |
| **Convergence** | Slow, unpredictable | Fast, directed |
| **Starting Point** | Single voice or mean | Super-seed (avg of 5) |
| **Evaluation** | Resemblyzer + features | WavLM + Whisper + quality |
| **Best For** | Quick experiments | Production quality |
| **Speed per Step** | Fast | Slower (more evaluation) |
| **Steps Needed** | 5,000-10,000 | 300-1,000 |
| **Total Time** | Medium | Medium-Long |

## Performance Tips

### 1. **Optimize for Speed During Development**

Use standard scoring during initial experiments:

```bash
python main.py \
  --target_audio ./test.wav \
  --target_text "Test text" \
  --use_cma_es \
  --step_limit 100
```

Then use advanced scoring for final run:

```bash
python main.py \
  --target_audio ./test.wav \
  --target_text "Test text" \
  --use_cma_es \
  --use_advanced_scoring \
  --step_limit 300
```

### 2. **Use Shorter Text During Optimization**

The code automatically truncates to 200 characters during CMA-ES iterations, but you can prepare shorter text:

```bash
--target_text "Short phrase for faster evaluation"
```

Then test the final voice with longer text.

### 3. **GPU Memory Management**

For long runs, the code automatically clears GPU cache every 10 iterations. If you encounter memory issues:

1. Reduce population size: `--population_limit 5`
2. Use smaller step limit and run multiple times
3. Monitor GPU usage: `nvidia-smi`

### 4. **Population Size**

The `--population_limit` controls how many top voices are used:

- Small (5-10): Faster initialization, less diversity
- Medium (10-15): Balanced (default: 10)
- Large (15-20): Slower initialization, more diversity

## Understanding the Output

### CMA-ES Progress Output

```
Iteration:50   Loss:0.8234 (CMA) | Trad Score:87.23 | Target Sim:0.891
```

- **Iteration**: Current CMA-ES iteration
- **Loss**: Composite loss (lower is better when using advanced scoring)
- **Trad Score**: Traditional Resemblyzer-based score (higher is better)
- **Target Sim**: Similarity to target voice (0-1, higher is better)

### Output Files

CMA-ES creates a timestamped directory with:

```
out/my_voice_target_cma_20231231_120000/
├── my_voice_0_0.8234.pt          # Best at iteration 0
├── my_voice_0_0.8234.wav
├── my_voice_45_0.7891.pt         # Improvement at iteration 45
├── my_voice_45_0.7891.wav
├── best_voice_final.pt           # Best overall result
└── best_voice_final.wav
```

## Troubleshooting

### Issue: "Advanced models not available"

**Solution:** Install required packages:
```bash
pip install transformers torchaudio jiwer
```

### Issue: "CMA-ES not available"

**Solution:** Install CMA:
```bash
pip install cma
```

### Issue: Out of memory during advanced scoring

**Solution:** 
1. Reduce population limit: `--population_limit 5`
2. Use CPU instead of GPU for models
3. Disable advanced scoring for initial runs

### Issue: Slow performance

**Solution:**
1. Don't use `--use_advanced_scoring` initially
2. Reduce `--step_limit` to 100-300 for CMA-ES
3. Use shorter target text during optimization
4. Consider using traditional random walk for quick tests

## Advanced Configuration

### Custom Quality Function

The current implementation uses a simple quality proxy. For production use, you can integrate NISQA or DNSMOS:

1. Install NISQA: https://github.com/gabrielmittag/NISQA
2. Modify `FitnessScorer._predict_quality_proxy()` in `utilities/fitness_scorer.py`
3. Replace the heuristic with actual NISQA prediction

### Hybrid Approach

Combine traditional and CMA-ES:

1. Run traditional random walk: `--step_limit 5000`
2. Use best result as starting point: `--starting_voice ./out/best_voice.pt`
3. Refine with CMA-ES: `--use_cma_es --use_advanced_scoring --step_limit 200`

## Algorithm Details

### Super-Seed Implementation

```python
# Gets top 5 voices by score
top_voices = sorted(all_voices, key=lambda x: x.score)[:5]

# Averages their vectors
mean_vector = torch.mean(torch.stack([v.tensor for v in top_voices]), dim=0)
```

### CMA-ES Process

```
1. Initialize population from super-seed (or mean)
2. Loop:
   a. ASK: CMA-ES generates candidate solutions
   b. EVALUATE: Score each candidate
   c. TELL: Report fitness back to CMA-ES
   d. CMA-ES updates distribution
3. Return best solution found
```

### Composite Scoring Formula

```python
final_score = (wavlm_similarity * 0.5) + (quality * 0.3) - (wer * 0.2)
```

Weights can be adjusted in `fitness_scorer.py` for your use case.

## Best Practices

1. **Start Simple**: Use basic random walk to understand your data
2. **Use Super-Seed**: Always enable for better initialization
3. **Progressive Refinement**: Run quick tests, then detailed optimization
4. **Save Checkpoints**: Keep intermediate results for comparison
5. **Validate Results**: Always listen to outputs before deploying

## Example Workflow

### Quick Test (5 minutes)
```bash
python main.py \
  --target_audio audio.wav \
  --target_text "Quick test" \
  --step_limit 100
```

### Production Quality (30-60 minutes)
```bash
python main.py \
  --target_audio audio.wav \
  --target_text "$(cat transcription.txt)" \
  --use_cma_es \
  --use_super_seed \
  --use_advanced_scoring \
  --step_limit 300 \
  --population_limit 10
```

## References

- CMA-ES: https://github.com/CMA-ES/pycma
- WavLM: https://huggingface.co/microsoft/wavlm-base-plus-sv
- Whisper: https://github.com/openai/whisper
- Original Paper: Hansen, N. (2016). "The CMA Evolution Strategy"

## Contributing

Found a bug or have a suggestion? Please open an issue on GitHub!

---

**Note**: This is an advanced feature set. For basic voice cloning, the traditional random walk approach (`python main.py --target_audio audio.wav --target_text "text"`) still works great and is much faster.
