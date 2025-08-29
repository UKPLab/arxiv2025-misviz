from torchvision.transforms import functional as F


class ResizeKeepAspectRatio:
    def __init__(self, size):
        self.size = size

    def __call__(self, img):
        w, h = img.size
        if w == h:
            return F.resize(img, (self.size, self.size))

        if w > h:
            new_w, new_h = self.size, int(self.size * h / w)
        else:
            new_w, new_h = int(self.size * w / h), self.size

        img = F.resize(img, (new_h, new_w))

        pad_w = max(0, self.size - new_w)
        pad_h = max(0, self.size - new_h)

        left = pad_w // 2
        top = pad_h // 2
        right = pad_w - left
        bottom = pad_h - top

        return F.pad(img, (left, top, right, bottom), fill=0)
