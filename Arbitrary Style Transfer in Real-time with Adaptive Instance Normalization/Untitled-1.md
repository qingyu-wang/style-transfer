python test.py --content input/content/cornell.jpg --style input/style/woman_with_hat_matisse.jpg


th train.lua -contentDir COCO_TRAIN_DIR -styleDir WIKIART_TRAIN_DIR

decoder-content-similar.t7  -styleWeight 1e-2 减小风格权重，更接近原图
decoder.t7                  -styleWeight 1e-1 增大风格权重，更突出风格



0.weight torch.Size([64, 3, 3, 3])
0.bias torch.Size([64])
2.weight torch.Size([64, 64, 3, 3])
2.bias torch.Size([64])
5.weight torch.Size([128, 64, 3, 3])
5.bias torch.Size([128])
7.weight torch.Size([128, 128, 3, 3])
7.bias torch.Size([128])
10.weight torch.Size([256, 128, 3, 3])
10.bias torch.Size([256])
12.weight torch.Size([256, 256, 3, 3])
12.bias torch.Size([256])
14.weight torch.Size([256, 256, 3, 3])
14.bias torch.Size([256])
16.weight torch.Size([256, 256, 3, 3])
16.bias torch.Size([256])
19.weight torch.Size([512, 256, 3, 3])
19.bias torch.Size([512])

m = {
    0: 2,
    2: 5,
    5: 9,
    7: 12,
    10: 16,
    12: 19,
    14: 22,
    16: 25,
    19: 29,
}

```python
import torch
import torch.nn as nn

import net

decoder = net.decoder
vgg = net.vgg
vgg.load_state_dict(torch.load("models/vgg_normalised.pth"))
vgg = nn.Sequential(*list(vgg.children())[:31])
network = net.Net(vgg, decoder)

torch.manual_seed(4)
a = torch.rand(4, 3, 224, 224)

network.encode(a)
```

```python
import torch
import torch.nn as nn

from model import AdaptiveInstanceNormModel

model = AdaptiveInstanceNormModel()
model.encoder.net.load_state_dict(torch.load("src/models/encoder.pth"))

torch.manual_seed(4)
a = torch.randn(4, 3, 224, 224)

model.encoder(a)
```


0.weight torch.Size([3, 3, 1, 1])
0.bias torch.Size([3])
2.weight torch.Size([64, 3, 3, 3])
2.bias torch.Size([64])
5.weight torch.Size([64, 64, 3, 3])
5.bias torch.Size([64])
9.weight torch.Size([128, 64, 3, 3])
9.bias torch.Size([128])
12.weight torch.Size([128, 128, 3, 3])
12.bias torch.Size([128])
16.weight torch.Size([256, 128, 3, 3])
16.bias torch.Size([256])
19.weight torch.Size([256, 256, 3, 3])
19.bias torch.Size([256])
22.weight torch.Size([256, 256, 3, 3])
22.bias torch.Size([256])
25.weight torch.Size([256, 256, 3, 3])
25.bias torch.Size([256])
29.weight torch.Size([512, 256, 3, 3])
29.bias torch.Size([512])
32.weight torch.Size([512, 512, 3, 3])
32.bias torch.Size([512])
35.weight torch.Size([512, 512, 3, 3])
35.bias torch.Size([512])
38.weight torch.Size([512, 512, 3, 3])
38.bias torch.Size([512])
42.weight torch.Size([512, 512, 3, 3])
42.bias torch.Size([512])
45.weight torch.Size([512, 512, 3, 3])
45.bias torch.Size([512])
48.weight torch.Size([512, 512, 3, 3])
48.bias torch.Size([512])
51.weight torch.Size([512, 512, 3, 3])
51.bias torch.Size([512])


```python
import glob
import os

import cv2
import numpy as np

src_paths = sorted(glob.glob("UV/CFD/*_src.jpg"))
mask_paths = sorted(glob.glob("UV/CFD/*_mask_all.jpg"))

print(len(src_paths))
print(len(mask_paths))

src_path = src_paths[0]
mask_path = mask_paths[0]

for src_path, mask_path in zip(src_paths, mask_paths):
    src = cv2.imread(src_path)
    mask = cv2.cvtColor(cv2.imread(mask_path), cv2.COLOR_BGR2GRAY)

    fg = cv2.bitwise_or(src, src, mask=mask)
    # cv2.imwrite("test_fg.jpg", fg)

    # mask_inv = cv2.bitwise_not(mask)
    bg = np.full(src.shape, 0, dtype=np.uint8)
    # bg = cv2.bitwise_or(bg, bg, mask=mask_inv)
    # cv2.imwrite("test_bg.jpg", bg)

    dst = cv2.bitwise_or(src, bg, mask=mask)
    save_path = "UV/HF/%s" % os.path.basename(src_path).replace("_src.jpg", ".jpg")
    cv2.imwrite(save_path, dst)

```

