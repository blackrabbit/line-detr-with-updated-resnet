from PIL import Image, ImageDraw

import torch.nn.functional as F
import torch

from models.letr import build
from models.misc import nested_tensor_from_tensor_list
from models.preprocessing import Compose, ToTensor, Resize, Normalize

import os

def create_letr(path):
    # obtain checkpoints
    checkpoint = torch.load(path, map_location='cpu')

    # load model
    args = checkpoint['args']
    args.device = 'cpu'
    model, _, _ = build(args)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    return model

def get_lines_and_draw(image, outputs, orig_size, threshold=.7):
    # find lines
    out_logits, out_line = outputs['pred_logits'], outputs['pred_lines']
    prob = F.softmax(out_logits, -1)
    scores, labels = prob[..., :-1].max(-1)
    img_h, img_w = orig_size.unbind(0)
    scale_fct = torch.unsqueeze(torch.stack(
        [img_w, img_h, img_w, img_h], dim=0), dim=0)
    lines = out_line * scale_fct[:, None, :]
    lines = lines.view(1000, 2, 2)
    lines = lines.flip([-1])  # this is yxyx format
    scores = scores.detach().numpy()
    keep = scores >= threshold
    keep = keep.squeeze()
    lines = lines[keep]
    if len(lines) != 0:
        lines = lines.reshape(lines.shape[0], -1)

        # draw lines
        draw = ImageDraw.Draw(image)
        for _, line in enumerate(lines):
            y1, x1, y2, x2 = line
            draw.line((x1, y1, x2, y2), fill=500, width=5)
    return lines

if __name__ == '__main__':
    model = create_letr('resnet50/checkpoint0024.pth')

    # Size of feature vector
    test_size = 1100
    normalize = Compose([
        ToTensor(),
        Normalize([0.538, 0.494, 0.453], [0.257, 0.263, 0.273]),
        Resize([test_size]),
    ])

    url = '/Users/blackrabbit/data/jags/outputvideos/11/'
    files = os.listdir(url)
    for i, x in enumerate(files):
        files[i] = url + x
    for image_url in files:
        image = Image.open(image_url)
        h, w = image.height, image.width
        orig_size = torch.as_tensor([int(h), int(w)])

        img = normalize(image)
        inputs = nested_tensor_from_tensor_list([img])

        with torch.no_grad():
            outputs = model(inputs)[0]
        lines = get_lines_and_draw(image, outputs, orig_size)

        file_name = os.path.basename(image_url)

        image.save('/tmp/letr_debug_images/' + file_name)

