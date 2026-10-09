# Image colorization U-Net weights

Trained weights for a small U-Net that turns a 256x256 grayscale image into a
color image. Trained on 30,000 COCO 2017 photos with an MSE loss.

| File | Format | Size | SHA-256 |
| --- | --- | --- | --- |
| `weights.pth` | PyTorch `state_dict` (140 tensors, 7.9 million numbers) | 31.7 MB | `5cc6a3f50111ffeedde42caf61b5ed7c3ebff87c8003fed85c417e3e54e7b457` |

Download: https://github.com/AmirShib/image-colorization-unet-weights/releases/download/v1.0/weights.pth

The file is only the numbers. It loads into the `ColorizeModel` class defined in
the accompanying notebook, which fetches this file by itself when `weights.pth`
is not already next to it:

```python
model = ColorizeModel()
model.load_state_dict(torch.load("weights.pth", map_location="cpu"))
model.eval()
```
