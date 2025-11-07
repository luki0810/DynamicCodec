from model.utils.class_choice.class_choices import ClassChoices

# module import
from model.quantizer.quantize import ResidualVectorQuantize, VectorQuantize
from model.quantizer.bsq import BinarySphericalQuantizer
from model.decoder.dac import Decoder as dac_Decoder
from model.encoder.dac import Encoder as dac_Encoder
from model.encoder.encodec import Encoder as encodec_Encoder
from model.decoder.encodec import Decoder as encodec_Decoder
from model.encoder.repcodec import Encoder as repcodec_Encoder

from model.utils.abs_class import AbsEncoder, AbsDecoder, AbsQuantizer


# choices set
encoder_choices = ClassChoices(
    name="encoder",
    classes=dict(
        default=dac_Encoder,
        dac=dac_Encoder,
        encodec=encodec_Encoder,
        repcodec=repcodec_Encoder
        # 你可以在这里再挂其它实现，比如：
        # conv=ConvEncoder, transformer=TransformerEncoder1D, ...
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
        bsq=BinarySphericalQuantizer
        # 也可添加：ema=EMAResidualVQ, gumbel=GumbelVQ, ...
    ),
    type_check=AbsQuantizer,
    default="default",
)

decoder_choices = ClassChoices(
    name="decoder",
    classes=dict(
        default=dac_Decoder,
        dac=dac_Decoder,
        encodec=encodec_Decoder
        # 再挂：big=BigDecoder, hifi=HiFiDecoder, ...
    ),
    type_check=AbsDecoder,
    default="default",
)