from model.utils.class_choice.class_choices import ClassChoices

# Quantizer import
from model.quantizer.quantize import ResidualVectorQuantize, VectorQuantize
from model.quantizer.bsq import BinarySphericalQuantizer
from model.quantizer.fsq import FiniteScalarQuantize
from model.quantizer.fsq import FiniteScalarQuantize


# Encoder and Decoder import
from model.encoder.dac import Encoder as dac_Encoder
from model.decoder.dac import Decoder as dac_Decoder
from model.encoder.encodec import Encoder as encodec_Encoder
from model.decoder.encodec import Decoder as encodec_Decoder
from model.encoder.repcodec import Encoder as repcodec_Encoder
from model.encoder.mel import Encoder as mel_Encoder
from model.decoder.mel import Decoder as mel_Decoder
# NOTE: cosmos uses Conv2d (image tokenizer); not yet adapted to 1D audio,
# so it's intentionally NOT registered in encoder_choices / decoder_choices.
# Source files kept in tree under model/{encoder,decoder}/cosmos.py for
# future 2D experiments; do not select it via conf/base.yaml until the
# adapter (e.g. mel-as-image reshape) is implemented in model/build.py.
# from model.encoder.cosmos import Encoder as cosmos_Encoder
# from model.decoder.cosmos import Decoder as cosmos_Decoder

# Vocoder import
from model.vocoder.voco_istft import Vocoder as Vocos

# Abstract class import
from model.utils.abs_class import AbsEncoder, AbsDecoder, AbsQuantizer, AbsVocoder


# choices set
encoder_choices = ClassChoices(
    name="encoder",
    classes=dict(
        default=dac_Encoder,
        dac=dac_Encoder,
        encodec=encodec_Encoder,
        repcodec=repcodec_Encoder,
        mel = mel_Encoder,
    ),
    type_check=AbsEncoder,
    default="default",
)

quantizer_choices = ClassChoices(
    name="quantizer",
    classes=dict(
        default=ResidualVectorQuantize,
        vq=VectorQuantize,
        rvq=ResidualVectorQuantize,
        bsq=BinarySphericalQuantizer,
        fsq=FiniteScalarQuantize,
    ),
    type_check=AbsQuantizer,
    default="default",
)

decoder_choices = ClassChoices(
    name="decoder",
    classes=dict(
        default=dac_Decoder,
        dac=dac_Decoder,
        encodec=encodec_Decoder,
        mel = mel_Decoder,
    ),
    type_check=AbsDecoder,
    default="default",
)

vocoder_choices = ClassChoices(
    name="vocoder",
    classes=dict(
        vocos = Vocos
    ),
    type_check=AbsVocoder,
    default=None,
)