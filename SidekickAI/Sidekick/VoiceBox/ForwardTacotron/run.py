# Load pretrained models
from notebook_utils.synthesize import (
    get_forward_model, get_melgan_model, get_wavernn_model, synthesize, init_hparams)
from utils import hparams as hp
from SidekickAI.Utilities import utils
import sounddevice as sd
init_hparams('notebook_utils/pretrained_hparams.py')
tts_model = get_forward_model('pretrained/forward_400K.pyt')
voc_melgan = get_melgan_model() 

# Synthesize with melgan
while True:
    input_text = input("> ")
    wav = synthesize(input_text, tts_model, voc_melgan, alpha=1)
    sd.play(wav, hp.sample_rate)
    sd.wait()