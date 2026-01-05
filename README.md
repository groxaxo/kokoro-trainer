# KVoiceWalk
KVoiceWalk creates new [Kokoro](https://github.com/hexgrad/kokoro) voice style tensors that clone target voices using advanced optimization algorithms and hybrid scoring methods. The project now supports both traditional random walk and professional CMA-ES optimization with advanced evaluation metrics, including **specialized support for Latin American Spanish voice training**.

This project is only possible because of the incredible work of projects like [Kokoro](https://github.com/hexgrad/kokoro) and [Resemblyzer](https://github.com/resemble-ai/Resemblyzer). I was struck by how small the Kokoro style tensors were and wondered if it would be possible to "evolve" new voice tensors more similar to target audio. The results are promising and this scoring method could be a valid option for a future genetic algorithm. I wanted more voice options for Kokoro, and now I have them.

## 🆕 NEW: Latin American Spanish Support! 🇪🇸

KVoiceWalk now includes **automatic detection and optimization** for Spanish language training:

- **🎯 Automatic Language Detection**: Recognizes Spanish text and enables Spanish-optimized parameters
- **📊 Spanish-Specific Scoring**: Rewards clear vowel pronunciation and syllable-timed rhythm
- **🔤 Text Normalization**: Handles Spanish abbreviations and accent marks properly  
- **🌎 Latin American Dialects**: Optimized for Mexican, Argentinian, Colombian, Caribbean, and other variants
- **📖 Comprehensive Guide**: See [SPANISH_GUIDE.md](SPANISH_GUIDE.md) for detailed instructions

### Quick Spanish Example:
```bash
uv run main.py \
  --target_audio ./mi_voz.wav \
  --target_text "Hola, ¿cómo estás? Me encanta hablar español." \
  --use_cma_es \
  --use_super_seed \
  --step_limit 500
```

When Spanish is detected, you'll see Spanish-optimized training recommendations automatically!

## 🆕 NEW: CMA-ES Optimization!

KVoiceWalk now supports professional-grade optimization with:

- **🎯 CMA-ES Algorithm**: Intelligent evolution strategy that learns the direction of improvement
- **🌟 Super-Seed Initialization**: Averages top 5 voices for robust starting points
- **🎤 Advanced Scoring**: WavLM for identity, Whisper for intelligibility, quality metrics
- **⚡ Faster Convergence**: Achieve better results in 300-1000 steps vs 5000-10000

See [CMA_ES_GUIDE.md](CMA_ES_GUIDE.md) for complete usage instructions.

### Quick CMA-ES Example:
```bash
uv run main.py \
  --target_audio ./example/target.wav \
  --target_text "Your transcription here" \
  --use_cma_es \
  --use_super_seed \
  --step_limit 500
```

## 🆕 Web UI with Gradio 6!

KVoiceWalk now includes a fully functional web interface built with Gradio 6! 

### Launch the Web UI:
```bash
uv run web_ui.py
```

For public sharing:
```bash
uv run web_ui.py --share
```

The Web UI includes:
- 🎵 **Voice Testing**: Test voice files with custom text
- 🔬 **Advanced Voice Cloning**: Full control over all parameters
- ⚡ **Quick Clone**: Automatic transcription + voice cloning in one step
- 📝 **Transcription**: Convert audio to text
- ❓ **Built-in Help**: Comprehensive usage guide

## Example Audio
#### Target Audio File (Generated Using A different text to speech library)

https://github.com/user-attachments/assets/ecafe828-4fee-41dd-8fe6-1f766ad19f56

#### The Most Similar Stock Trained Voice From Kokoro, af_heart.pt. Similarity score of 71%

https://github.com/user-attachments/assets/0d693c60-a9f4-43bc-bb36-409bd7391d79

#### KVoiceWalk Generated Voice Tensor After 10,000 steps. Similarity score of 93% (From Resemblyzer)

https://github.com/user-attachments/assets/b19559dd-712c-427d-8ec4-93ff26daaa92

## Installation
1. Clone this repository, change directory into it, setup environment, install dependencies
```bash
git clone https://github.com/RobViren/kvoicewalk.git
cd kvoicewalk
uv venv --python 3.10
source .venv/bin/activate # '.venv\Scripts\activate' if Windows
uv sync
```

## Quick Start with Web UI (Recommended)
2. Launch the Gradio web interface:
```bash
uv run web_ui.py
```

Open your browser to `http://localhost:7860` and enjoy the intuitive interface!

## Command Line Usage
2. KVoiceWalk expects target audio files to be in Mono 24000 Hz sample rate wav file format; ideally 20-30 seconds of a single speaker. However if needed, Kvoicewalk will check and convert target audio files into the proper format. If you would prefer to prepare them beforehand, you can this use this example ffmpeg command. 

```bash
ffmpeg -i input_file.wav -ar 24000 target.wav
```

3. Use [uv](https://docs.astral.sh/uv/) to run the application with arguments.

```bash
uv run main.py --target_text "The old lighthouse keeper never imagined that one day he'd be guiding ships from the comfort of his living room, but with modern technology and an array of cameras, he did just that, sipping tea while the storm raged outside and gulls shrieked overhead." --target_audio ./example/target.wav
```

4. KVoiceWalk will now go through each voice in the voices folder to find the closest matches to the target file. After narrowing that down using the **population_limit** argument, it will begin to randomly guess and check voices keeping the best voice as the source for the random walk. It will log the progress and save audio and voice tensors to the **out** folder. You can then use these voice tensors in your other projects or generate some audio using the following command.

```bash
uv run main.py --test_voice /path/to/voice.pt --target_text "Your really awesome text you want spoken"
```

This will generate an audio file called out.wav using the supplied *.pt file you give it. This way you can easily test a variety of voice tensors and input text.

Play with the command line arguments and find what works for you. This is a fairly random process and processing for a long time could suddenly result in a better outcome. You can create a folder of your favorite sounding voices from the other random walks and use that as the basis for interpolation or just use that as the source for the next random walk. You can pass **starting_voice** argument to tell the system exactly what to use as a base if you want. Playing around with the options can get you a voice closer to the style of the target.

## CMA-ES Optimization (NEW!)

For production-quality results with faster convergence, use the CMA-ES optimization:

### Basic CMA-ES Usage
```bash
uv run main.py \
  --target_text "Your transcription here" \
  --target_audio ./example/target.wav \
  --use_cma_es \
  --step_limit 500
```

### CMA-ES with Super-Seed (Recommended)
```bash
uv run main.py \
  --target_text "Your transcription here" \
  --target_audio ./example/target.wav \
  --use_cma_es \
  --use_super_seed \
  --step_limit 300
```

### CMA-ES with Advanced Scoring (Best Quality)
```bash
uv run main.py \
  --target_text "Your transcription here" \
  --target_audio ./example/target.wav \
  --use_cma_es \
  --use_super_seed \
  --use_advanced_scoring \
  --step_limit 300
```

**Features:**
- **Super-Seed**: Averages top 5 voices for better initialization
- **CMA-ES**: Professional optimization algorithm (faster convergence)
- **Advanced Scoring**: Uses WavLM (identity) + Whisper (intelligibility) + quality metrics

**Performance:**
- Traditional random walk: 5,000-10,000 steps needed
- CMA-ES: 300-1,000 steps for better results
- Advanced scoring is slower per step but achieves better quality

See [CMA_ES_GUIDE.md](CMA_ES_GUIDE.md) for detailed documentation and examples.

## Interpolated Start
KVoiceWalk has a function to interpolate around the trained voices and determine the best possible starting population of tensors to act as a guide for the random walk function to clone the target voice. Simply run the application as follows to run interpolation first. This does take awhile and having a beefy GPU will help with processing time.

```bash
uv run main.py --target_text "The works the speaker says in the audio clip" --target_audio /path/to/target.wav --interpolate_start
```

This will run an interpolation search for the best voices and put them in a folder labeled **interpolated** which you can use as the basis for a new random walk later. It will also continue a random walk afterwards.

## Example Outputs
The closest voice in the trained models for the example/target.wav was af_heart.pt with the following stats.
```
af_heart.pt          Target Sim: 0.709, Self Sim: 0.978, Feature Sim: 0.47, Score: 81.22
```
Interpolation search gave a voice that had the following stats.
```
af_jessica.pt_if_sara.pt_0.10.pt Target Sim: 0.780, Self Sim: 0.973, Feature Sim: 0.34, Score: 84.20
```
The interpolation showed a big improvement. The population of interpolated voices is then used as the basis for standard deviation mutation of a supplied voice tensor. After 10,000 steps of random walking and replacing with the best, we get this.
```
Step:9371, Target Sim:0.917, Self Sim:0.971, Feature Sim:0.54, Score:92.99, Diversity:0.01
```
An improvement of 13.7% in similarity while still maintaining model stability and voice quality.

## Design
By far the hardest thing to get right was the scoring function. Earlier attempts using Resemblyzer only resulted in overfitted garbage. Self similarity was important in keeping the model producing the same sounding input despite different inputs. Self similarity represented stability in the model and was critical in evaluation.

But even with self similarity and similarity presented by Resemblyzer it was not enough. I had to add an audio feature similarity comparison in order to prevent audio quality getting poor. What happened without this is the audio would pass similarity and self similarity checks but again sound like a metal basket of tools being thrown down stairs. The feature comparison made the difference and prevented over fitting to a random sound that apparently sounded similar to the target wav file.

The other secret sauce was the harmonic mean calculation that controls the scoring. The harmonic mean allows for some backsliding on self similarity, feature similarity, and target similarity so long as the improvement goes the right way. This made exploring the space easier for the system instead of requiring that all three only improve, which led to quick and sad stagnation. I lowered the weighting on the feature similarity. I mainly need that to prevent the voice from going completely out of bounds.

## Notes
This does not run in parallel, but does adopt early returning on bad tensors. You can run multiple instances assuming you have the GPU/CPU for it. I can run about 2 in parallel on my 3070 laptop. The results are random. You can have some that led to incredible sounding results after stagnating for a long time, and others can crash and burn right away. Totally random. This is where a future genetic algorithm would be better. But the random walk proves out the theory.

Other things you could do:
- Populate a database with results from this and train a model to predict similarity and see if you can use that to more tightly guide voice creation
- Use different methods for voice generation than my simple method, though PCA had some challenges
- Implement your own genetic algorithm and evolve voice tensors instead of random walk

## KVoiceWalk Features

## Transcribe Start, --transcribe_start
KvoiceWalk con use Faster-Whisper to quickly convert your audio clip to text and update your --target_text. A copy of the transcription is also saved as a txt file in the ./texts folder. Txt files can be used as the --target-text argument with a relative path /path/to/your/transcribed.txt. This can be combined with --interpolate_start also.

```bash
uv run main.py --target_text "This text will be replaced!" --target_audio /path/to/target.wav --transcribe_start

uv run main.py --target_text /path/to/your/transcribed.txt --target_audio /path/to/target.wav
```

## Transcribe Many, --transcribe_many
KVoiceWalk can be used for file prep prior to your runs. With --transcribe_many, single file wav or folders containing wav files may be transcribed and their transcriptions saved as individual txt files in the ./texts folder.

```bash
uv run main.py --target_audio /path/to/target.wav --transcribe_many

uv run main.py --target_audio /path/to/audio/Folder/ --transcribe_many
```

## Export Voices, --voices_folder and --export_bin
Voices with a folder can be exported by passing --voices_folder and --export_bin in the command line. All .pt voices in the --voices_folder /path/to/your/voices/ argument will be packaged together as 'voices.bin' and saved in the same folder.

```bash
uv run main.py --voices_folder ./voices --export_bin
```

## All KVoiceWalk Arguments
```
## General Arguments
"--target_text", type=str, help="The words contained in the target audio file.
    Should be around 100-200 tokens (two sentences). Alternatively, can point to a txt file of the transcription."

"--other_text", type=str, help="A segment of text used to compare self similarity. Should be around 100-200 tokens." 
    default="If you mix vinegar, baking soda, and a bit of dish soap in a tall cylinder, the resulting eruption is both
    a visual and tactile delight, often used in classrooms to simulate volcanic activity on a miniature scale."

"--voice_folder", type=str, help="Path to the voices you want to use as part of the random walk.", default="./voices"

"--transcribe_start", help="Input: filepath to wav file Output: Transcription .txt in ./texts Transcribes a target wav or wav folder and replaces --target_text''

"--interpolate_start", help="Goes through an interpolation search step before random walking", action='store_true'

"--population_limit", type=int, help="Limits the amount of voices used as part of the random walk", default=10

"--step_limit", type=int, help="Limits the amount of steps in the random walk", default=10000)

"--output_name", type=str, help="Filename for the generated output audio", default="out.wav")

## Arguments for Random Walk mode
"--target_audio", type=str, help="Path to the target audio file. Must be 24000 Hz mono wav file."

"--starting_voice", type=str, help="Path to the starting voice tensor"

"--use_super_seed", help="Use super-seed initialization (average of top 5 voices)", action='store_true'

"--use_cma_es", help="Use CMA-ES optimization instead of random walk", action='store_true'

"--cma_sigma", type=float, help="CMA-ES initial step size (default: 0.1)", default=0.1

"--use_advanced_scoring", help="Use WavLM/Whisper scoring (requires transformers, jiwer)", action='store_true'

"--lang_code", type=str, help="Language code for TTS (a=auto-detect, es=Spanish, en=English)", default="a"

## Arguments for Test mode
"--test_voice", type=str, help="Path to the voice tensor you want to test"

## Arguments for Util mode
"--export_bin", help='Exports target voices in the --voice_folder directory', action='store_true'

"--transcribe_many", help='Input: filepath to wav file or folder\nOutput: Individualized transcriptions in ./texts folder\nTranscribes a target wav or wav folder. Replaces --target_text'
