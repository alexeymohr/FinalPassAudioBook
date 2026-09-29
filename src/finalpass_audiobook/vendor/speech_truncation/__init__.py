try:
    from .configuration_speech_truncation_detection import SpeechTruncationDetectionConfig
    from .modeling_speech_truncation_detection import (
        SpeechTruncationDetectionModel,
        SpeechTruncationPrediction,
    )
    from .processing_speech_truncation_detection import SpeechTruncationDetectionProcessor
except ImportError:
    # Fallback for direct module loading where package-relative imports are unavailable.
    from configuration_speech_truncation_detection import SpeechTruncationDetectionConfig
    from modeling_speech_truncation_detection import (
        SpeechTruncationDetectionModel,
        SpeechTruncationPrediction,
    )
    from processing_speech_truncation_detection import SpeechTruncationDetectionProcessor

__all__ = [
    "SpeechTruncationDetectionConfig",
    "SpeechTruncationDetectionModel",
    "SpeechTruncationPrediction",
    "SpeechTruncationDetectionProcessor",
]
