import argparse
import pickle 
import os
import time

import cv2
import imageio
import matplotlib.pyplot as plt
import numpy as np 
import skimage
import torch
from torchvision import transforms 
from PIL import Image


from netdissect.broden import BrodenDataset
import netdissect.dissection as dissection

from build_vocab import Vocabulary
from data_loader import get_loader 
from model import EncoderCNN, DecoderRNN


PARENT_DIR = os.path.dirname(__file__)
SIZE = (224, 224)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def extract(args):
    # Image preprocessing
    transform = transforms.Compose([
        transforms.Resize(SIZE),
        transforms.ToTensor(), 
        transforms.Normalize((0.485, 0.456, 0.406), 
                             (0.229, 0.224, 0.225))])
    
    # Load vocabulary wrapper
    with open(args.vocab_path, 'rb') as f:
        vocab = pickle.load(f)

    # Build models
    encoder = EncoderCNN(args.embed_size).eval()  # eval mode (batchnorm uses moving mean/variance)
    # decoder = DecoderRNN(args.embed_size, args.hidden_size, len(vocab), args.num_layers)

    dissection.retain_layers(encoder, [
        ('resnet.7.2.relu', 'final_layer'),
    ])

    encoder = encoder.to(device)
    # decoder = decoder.to(device)

    # Load the trained model parameters
    encoder.load_state_dict(torch.load(args.encoder_path))
    # decoder.load_state_dict(torch.load(args.decoder_path))

    encoder.eval()
    encoder = encoder.to(device)
    # decoder = decoder.to(device)

    # Load the trained model parameters
    encoder.load_state_dict(torch.load(args.encoder_path))
    # decoder.load_state_dict(torch.load(args.decoder_path))

    # Load data
    data_loader = get_loader(args.image_dir, args.caption_path, vocab,
                             transform, args.batch_size,
                             shuffle=False, num_workers=args.num_workers)
    
    # Run the models
    with torch.no_grad():
        total_step = len(data_loader)
        os.makedirs(os.path.join(PARENT_DIR, 'results', 'activations'), exist_ok=True)
        path = os.path.join(PARENT_DIR, 'results', 'samples.txt')
        with open(path, 'w') as results_file:
            start = time.time()
            for batch, (images, captions, lengths) in enumerate(data_loader):
                
                # Set mini-batch dataset
                images = images.to(device)
                # captions = captions.to(device)
                # targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]
                
                # Forward, backward and optimize
                features = encoder(images)
                # outputs = decoder(features, captions, lengths)
                # loss = criterion(outputs, targets)
                # decoder.zero_grad()
                # encoder.zero_grad()
                # loss.backward()
                # optimizer.step()

                activations = encoder.retained['final_layer']

                images = dissection.ReverseNormalize((0.485, 0.456, 0.406), 
                                                  (0.229, 0.224, 0.225))(images)
                images = images.cpu().numpy().transpose([0,2,3,1])
                activations = activations.cpu().numpy()

                scores = np.max(activations, axis=(-1,-2))
                samples = np.argmax(scores, axis=-1)
                gathered = activations[np.arange(len(samples)), samples].transpose([1,2,0])
                mask = cv2.resize(gathered, SIZE).transpose([2,0,1])
                k = int(0.8 * mask.size)
                threshhold = np.partition(mask, k, axis=None)[k]
                mask = mask >= threshhold
                mask = np.expand_dims(mask, axis=-1)
                outimg = np.concatenate((images, (1+mask)/2.), axis=-1)
                # outimg = outimg * mask
                activations = outimg

                for i, sample in enumerate(samples):
                    i += args.batch_size * batch
                    results_file.write('{} {}\n'.format(i, sample))
                for i, activation in enumerate(activations):
                    i += args.batch_size * batch
                    path = os.path.join(PARENT_DIR, 'results', 'activations', '{}.png'.format(i))
                    outactivation = skimage.img_as_ubyte(activation)
                    imageio.imwrite(path, outactivation)
                clock = time.time()
                delay = clock - start
                start = clock
                max_batch = 100
                # print('Step {}/{}: Time = {:.2f}'.format(batch, len(data_loader), delay))
                print('Step {}/{}: Time = {:.2f}'.format(batch, max_batch, delay))
                if batch == max_batch:
                    break


def main(args):
    extract(args)

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, help='input image for generating caption')
    parser.add_argument('--encoder_path', type=str, default='models/encoder-5-3000.ckpt', help='path for trained encoder')
    parser.add_argument('--decoder_path', type=str, default='models/decoder-5-3000.ckpt', help='path for trained decoder')
    parser.add_argument('--vocab_path', type=str, default='data/vocab.pkl', help='path for vocabulary wrapper')
    
    # Model parameters (should be same as paramters in train.py)
    parser.add_argument('--embed_size', type=int , default=256, help='dimension of word embedding vectors')
    parser.add_argument('--hidden_size', type=int , default=512, help='dimension of lstm hidden states')
    parser.add_argument('--num_layers', type=int , default=1, help='number of layers in lstm')

    parser.add_argument('--image_dir', type=str, default='data/train2014', help='directory for images')
    parser.add_argument('--caption_path', type=str, default='data/annotations/captions_train2014.json', help='path for train annotation json file')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=2)

    args = parser.parse_args()
    main(args)
