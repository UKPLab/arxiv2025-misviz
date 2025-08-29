import os
import sys
from abc import ABC, abstractmethod

root_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_folder not in sys.path:
    sys.path.insert(0, root_folder)
import image_processors


class ImageEncodingStrategy(ABC):
    def __init__(self):
        self._create_processor()
        self.encoder = None

    def prepare_encoder(self, backend):
        if not self.encoder:
            self._initialize_encoder()
        self.encoder.to(backend)

        for param in self.encoder.parameters():
            param.requires_grad = False

    def set_processor_augmentations(self, augmentations):
        self.processor.set_augmentations(augmentations)

    def preprocess_images(self, image):
        return self.processor.process_image(image)

    def encode(self, input_data):
        if not self.encoder:
            raise ValueError(
                "Encoder not initialized. Call prepare_encoder() first to initialize the encoder."
            )
        return self.encoder(input_data)

    def get_embedding_from_output(self, output):
        return output.last_hidden_state.mean(dim=1)

    @abstractmethod
    def _create_processor(self):
        pass

    @abstractmethod
    def _initialize_encoder(self):
        pass

    @abstractmethod
    def get_name(self):
        pass


class TinyChartEncodingStrategyBase(ImageEncodingStrategy):
    def _create_processor(self):
        self.processor = image_processors.TinyChartProcessor([])