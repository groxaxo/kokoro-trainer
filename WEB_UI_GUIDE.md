# Web UI Quick Start Guide

## 🚀 Launch the Web Interface

```bash
# Basic launch (local only)
python web_ui.py

# With public sharing link
python web_ui.py --share

# Custom port
python web_ui.py --port 8080

# Public server (accessible from network)
python web_ui.py --server-name 0.0.0.0 --port 7860
```

Then open your browser to: `http://localhost:7860`

## 📚 Tab Overview

### 🎵 Test Voice
**Purpose**: Test existing voice files with custom text

**Steps**:
1. Upload a `.pt` voice file
2. Enter text you want to synthesize
3. Adjust speech speed (0.5x to 2.0x)
4. Click "Generate Speech"
5. Listen to the result and download if needed

**Use Case**: Quick testing of voice files before using them in projects

---

### 🔬 Clone Voice (Advanced)
**Purpose**: Create a new voice clone with full control

**Steps**:
1. Upload target audio (20-30 seconds of clean speech)
2. Enter the **exact text** spoken in the audio
3. Enter "other text" for self-similarity testing (optional)
4. Configure settings:
   - **Population Limit**: Number of base voices to use (10 recommended)
   - **Step Limit**: Training iterations (higher = better, slower)
   - **Interpolation**: Better starting point (recommended but slower)
5. Click "Start Cloning"
6. Wait for completion (can take minutes to hours depending on settings)
7. Download best voice file and audio

**Settings Guide**:
- **Quick Test**: Population=5, Steps=1000, No interpolation
- **Standard**: Population=10, Steps=10000, No interpolation  
- **High Quality**: Population=10, Steps=20000, With interpolation
- **Maximum**: Population=20, Steps=50000, With interpolation

**Tips**:
- Use high-quality, clear audio
- Speak naturally in target audio
- Match the text exactly to what's spoken
- Longer step limits = better results but slower

---

### ⚡ Quick Clone
**Purpose**: Fastest way to clone a voice with automatic transcription

**Steps**:
1. Upload target audio
2. Adjust settings if needed (defaults work well)
3. Click "Quick Clone"
4. The system will:
   - Automatically transcribe the audio
   - Display the transcription
   - Start voice cloning
   - Return the best result

**When to Use**:
- You don't want to manually transcribe
- You want quick results
- You're testing different audio samples

---

### 📝 Transcribe
**Purpose**: Convert audio to text

**Steps**:
1. Upload audio file (any format)
2. Click "Transcribe"
3. Copy the transcription text

**Use Cases**:
- Get transcription for manual voice cloning
- Verify what was said in audio
- Prepare text for voice testing

---

## 🎯 Common Workflows

### Workflow 1: Test Existing Voices
```
Test Voice Tab → Upload voice.pt → Enter text → Generate → Listen
```

### Workflow 2: Quick Voice Clone
```
Quick Clone Tab → Upload audio → Click Quick Clone → Wait → Download
```

### Workflow 3: High-Quality Voice Clone
```
Transcribe Tab → Upload audio → Get text
↓
Clone Voice Tab → Upload same audio → Paste text
↓
Configure: Population=10, Steps=20000, Interpolation=Yes
↓
Start Cloning → Wait → Download best results
```

### Workflow 4: Voice Experimentation
```
Clone Voice Tab → Try different population/step combinations
↓
Compare results in output folder
↓
Use best voice in Test Voice tab
```

## 📁 Output Files

Results are saved in: `./out/[output_name]_[target_name]_[timestamp]/`

Files include:
- `*.wav` - Generated audio samples
- `*.pt` - Voice tensor files
- Filenames contain scores: `[name]_[step]_[score]_[similarity]_[target].wav`

**File Naming**:
- Higher score = better overall quality
- Higher similarity = closer to target voice
- Latest step ≠ always best (algorithm explores)

## ⚙️ Advanced Settings

### Population Limit
- **What**: Number of top-performing base voices to use
- **Range**: 1-50
- **Default**: 10
- **Impact**: Higher = more diverse exploration, slower
- **Recommendation**: 10-15 for most cases

### Step Limit
- **What**: Number of random walk iterations
- **Range**: 100-50000
- **Default**: 10000
- **Impact**: Higher = better results, much slower
- **Recommendation**: 
  - 1000 = quick test
  - 10000 = standard
  - 20000+ = high quality

### Interpolation Start
- **What**: Pre-search for optimal starting voices
- **When**: Recommended for best results
- **Cost**: Adds significant time upfront
- **Benefit**: Better starting point = better final result

### Starting Voice
- **What**: Specific voice file to start from (optional)
- **When**: You have a voice similar to target
- **How**: Upload `.pt` file
- **Benefit**: Faster convergence to target

## 🐛 Troubleshooting

### "Failed to load voice file"
- Ensure file is a valid `.pt` PyTorch tensor
- Try with a different voice file
- Check file isn't corrupted

### "Transcription failed"
- Ensure audio is clear and contains speech
- Try converting to WAV format first
- Check audio isn't corrupted or too quiet

### "Voice cloning stuck at 0%"
- This is normal for initialization
- Wait 30-60 seconds for processing to start
- Check console for error messages

### Out of Memory
- Reduce step limit
- Reduce population limit
- Close other applications
- Use GPU if available

### Results sound bad
- Try higher step limit
- Enable interpolation start
- Use clearer target audio
- Ensure text matches audio exactly

## 💡 Tips for Best Results

### Audio Quality
✅ **Do**:
- Use 20-30 seconds of clear speech
- Single speaker only
- Minimal background noise
- Natural speaking pace
- Consistent volume

❌ **Don't**:
- Multiple speakers
- Music or loud background
- Very short clips (< 10 seconds)
- Shouting or whispering
- Heavy compression artifacts

### Text Accuracy
✅ **Do**:
- Match exactly what's spoken
- Include punctuation
- Use complete sentences
- Transcribe carefully

❌ **Don't**:
- Approximate the words
- Add extra text
- Remove filler words if they're spoken
- Use different language

### Parameter Selection
✅ **Do**:
- Start with defaults
- Increase steps for better quality
- Use interpolation for final versions
- Experiment with population

❌ **Don't**:
- Max everything (too slow)
- Use step=100 for final (too few)
- Skip interpolation for best quality
- Use population=1 (too narrow)

## 📊 Expected Processing Times

**On CPU** (approximate):
- Quick Clone: 30-60 minutes
- Standard Clone: 2-4 hours  
- High Quality: 6-12 hours

**On GPU** (approximate):
- Quick Clone: 10-20 minutes
- Standard Clone: 30-60 minutes
- High Quality: 1-3 hours

*Times vary based on hardware, population, and step count*

## 🎓 Learning Path

1. **Start**: Test existing voices (Test Voice tab)
2. **Experiment**: Try Quick Clone with sample audio
3. **Learn**: Use Advanced Clone with low steps
4. **Master**: High-quality clones with optimization
5. **Create**: Your own voice library!

## 🔗 Additional Resources

- Main README: `README.md`
- Improvements: `IMPROVEMENTS.md`
- Code Examples: `main.py`
- Voice Files: `./voices/` directory

## 🆘 Support

If you encounter issues:
1. Check this guide
2. Review console output for errors
3. Try with default settings
4. Ensure dependencies are installed
5. Check GitHub issues

---

**Happy Voice Cloning! 🎙️**
