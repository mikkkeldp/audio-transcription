### Audio transcription with diarisation

The audio was transcribed using Whispe. Speech segments were embedded and clustered to label each segment with the corresponding speaker for accurate diarization. 

##### Setup
1. Install requirements
```
conda env create -f path/to/environment.yml
```

2. Run transcription. Sample audio inputs can be found in `/audio` and script output transcription will can bels under `/transcriptions`. 
```
python audio_transcribe.py
```