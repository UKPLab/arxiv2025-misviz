from PIL import Image

from transformers import (
    Pix2StructImageProcessor,
    DonutProcessor,
    AutoImageProcessor,
)
from abc import abstractmethod
from torchvision.transforms import v2 as transforms

import custom_transforms

from tinychart.model.builder import load_pretrained_model


def build_transform(augmentations):
    transform_list = []

    if "rotation" in augmentations:
        transform_list.append(transforms.RandomRotation(3))

    if "perspective" in augmentations:
        transform_list.append(transforms.RandomPerspective(0.2))

    return transform_list


class ImageProcessor:
    def __init__(self, augmentations):
        self.set_augmentations(augmentations)

    @abstractmethod
    def process_image(self, image):
        pass

    def set_augmentations(self, augmentations):
        if augmentations:
            self.transforms = transforms.Compose(
                build_transform(augmentations),
            )
        else:
            self.transforms = None


class Pix2StructProcessor(ImageProcessor):
    def __init__(self, model_type, augmentations, header_text):
        super().__init__(augmentations)
        self.processor = Pix2StructImageProcessor.from_pretrained(model_type)
        self.header_text = header_text

    def process_image(self, image):
        if self.transforms:
            image = self.transforms(image)
        processed_image = self.processor(
            image,
            header_text=self.header_text,
            return_tensors="pt",
        )
        flattened_patches = processed_image.flattened_patches
        return flattened_patches


class TinyChartProcessor(ImageProcessor):
    def __init__(self, augmentations):
        super().__init__(augmentations)
        model_path = "mPLUG/TinyChart-3B-768"
        _, self.model, self.processor, _ = load_pretrained_model(
            model_path,
            model_base=None,
            model_name="TinyChart-3B-768",
            device="cpu",
        )

    def get_model(self):
        return self.model

    def process_image(self, image):
        image = self.expand2square(
            image, tuple(int(x * 255) for x in self.processor.image_mean)
        )
        if self.transforms:
            image = self.transforms(image)
        preprocessed_image = self.processor.preprocess(image, return_tensors="pt")[
            "pixel_values"
        ]
        return preprocessed_image

    def expand2square(self, pil_img, background_color):
        width, height = pil_img.size
        if width == height:
            return pil_img
        elif width > height:
            result = Image.new(pil_img.mode, (width, width), background_color)
            result.paste(pil_img, (0, (width - height) // 2))
            return result
        else:
            result = Image.new(pil_img.mode, (height, height), background_color)
            result.paste(pil_img, ((height - width) // 2, 0))
            return result


class UnichartProcessor(ImageProcessor):
    def __init__(self, augmentations):
        super().__init__(augmentations)
        self.processor = DonutProcessor.from_pretrained(
            "ahmed-masry/unichart-chartqa-960"
        )

    def process_image(self, image):
        if self.transforms:
            image = self.transforms(image)
        processed_image = self.processor(image, return_tensors="pt")
        pixel_values = processed_image.pixel_values
        return pixel_values


class SiglipProcessor(ImageProcessor):
    def __init__(self, model_type, augmentations):
        self.processor = AutoImageProcessor.from_pretrained(model_type)
        super().__init__(augmentations)

    def process_image(self, image):
        if self.transforms:
            image = self.transforms(image)
        image_encoded = self.processor(image, return_tensors="pt")
        return image_encoded.pixel_values

    def set_augmentations(self, augmentations):
        size = self.processor.size["height"]
        adaptable_transformations = build_transform(augmentations)
        all_transforms = [custom_transforms.ResizeKeepAspectRatio(size)]
        if adaptable_transformations:
            all_transforms.extend(adaptable_transformations)

        self.transforms = transforms.Compose(all_transforms)


class ClipProcessor(ImageProcessor):
    def __init__(self, model_type, augmentations):
        self.processor = AutoImageProcessor.from_pretrained(model_type)
        self.set_augmentations(augmentations)

    def process_image(self, image):
        image = self.transforms(image)
        image_encoded = self.processor(image, return_tensors="pt")
        return image_encoded.pixel_values

    def set_augmentations(self, augmentations):
        size = self.processor.crop_size["height"]

        adaptable_transformations = build_transform(augmentations)
        all_transforms = [
            custom_transforms.ResizeKeepAspectRatio(size),
        ]
        if adaptable_transformations:
            all_transforms.extend(adaptable_transformations)

        self.transforms = transforms.Compose(all_transforms)
