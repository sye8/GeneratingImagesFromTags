import os
import pickle
import random

from torchvision import transforms

from dataset import *


if __name__ == "__main__":
    SUBSET_SIZE = 4800
    t = [transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), transforms.ToTensor()]
    t = transforms.Compose(t)
    train_set = get_COCO_captions_2014_train("MS-COCO", t, DALLE_TEXT_SEQ_LEN)
    train_loader = DataLoader(train_set, batch_size=SUBSET_SIZE, shuffle=True)
    img_batch, tag_batch = next(iter(train_loader))
    with open(os.path.join("datasubsets", "COCO_img_cap_subset_4800_imgtensor.pkl"), 'wb') as f:
        pickle.dump(img_batch, f)
    with open(os.path.join("datasubsets", "COCO_img_cap_subset_4800_tagtensor.pkl"), 'wb') as f:
        pickle.dump(tag_batch, f)
