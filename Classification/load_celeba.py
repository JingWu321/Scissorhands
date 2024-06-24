import torch
from torch.utils.data import Dataset
import os
from PIL import Image


class CelebA(Dataset):

    def __init__(self, root, train=True, transform=None, identity=None):

        self.root = root
        self.transform = transform
        self.identity = identity

        if train:
            ann_path = self.root + '/train.txt'
        else:
            ann_path = self.root + '/test.txt'

        images = []
        # targets = []
        identities = []
        for line in open(ann_path, 'r'):
            sample = line.split()
            if len(sample) != 42:
                raise(RuntimeError('Annotated face attributes of CelebA dataset should not be different from 40'))
            if self.identity is None or int(sample[1]) in self.identity:
                images.append(sample[0])
                identities.append(int(sample[1]))
                # targets.append([int(i) for i in sample[2:]])
            else:
                continue

        self.data = [os.path.join(self.root, 'img_align_celeba', img) for img in images]
        # self.targets = targets
        self.identities = identities
        # attr_cls = [
        #     '5_o_Clock_Shadow','Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes', \
        #     'Bald', 'Bangs','Big_Lips', 'Big_Nose', 'Black_Hair', 'Blond_Hair', \
        #     'Blurry', 'Brown_Hair', 'Bushy_Eyebrows', 'Chubby', 'Double_Chin', \
        #     'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones', \
        #     'Male', 'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard', \
        #     'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline', 'Rosy_Cheeks', \
        #     'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair', 'Wearing_Earrings', \
        #     'Wearing_Hat', 'Wearing_Lipstick', 'Wearing_Necklace', 'Wearing_Necktie', 'Young'
        #     ]


    def __len__(self):

        return len(self.data)

    def __getitem__(self, index):

        # Load data and get label
        img = Image.open(self.data[index]).convert('RGB')
        identities = self.identities[index]

        if self.transform is not None:
            img = self.transform(img)

        return img, identities



