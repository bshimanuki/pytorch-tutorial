import torch
import matplotlib.pyplot as plt
import numpy as np 
import argparse
import pickle 
import os
from torchvision import transforms 
from PIL import Image


from netdissect.broden import BrodenDataset
import netdissect.dissection as dissection

from build_vocab import Vocabulary
from data_loader import get_loader 
from model import EncoderCNN, DecoderRNN


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_image(image_path, transform=None):
    if image_path is None:
        image = Image.fromarray(np.zeros((1,1,3), dtype=np.uint8))
    else:
        image = Image.open(image_path)
    image = image.resize([224, 224], Image.LANCZOS)
    
    if transform is not None:
        image = transform(image).unsqueeze(0)
    
    return image

def get_covariance(args):
    # Image preprocessing
    transform = transforms.Compose([
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
        ('resnet.7.2.bn3', 'final_layer'),
    ])

    encoder = encoder.to(device)
    # decoder = decoder.to(device)

    # Load the trained model parameters
    encoder.load_state_dict(torch.load(args.encoder_path))
    # decoder.load_state_dict(torch.load(args.decoder_path))

    bds = BrodenDataset('data',
                        transform_image=transform,
                        resolution=224,
                        size=10000)
    data_loader = torch.utils.data.DataLoader(
        bds,
        batch_size=8, num_workers=2,
        # pin_memory=(device.type == 'cuda'),
    )
    # data_loader = get_loader(args.image_dir, args.caption_path, vocab,
                             # transform, args.batch_size,
                             # shuffle=False, num_workers=args.num_workers)

    outdir = 'covariance'
    dissection.collect_covariance(outdir, encoder, data_loader)


def sample(args):
    # Image preprocessing
    transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize((0.485, 0.456, 0.406), 
                             (0.229, 0.224, 0.225))])
    
    # Load vocabulary wrapper
    with open(args.vocab_path, 'rb') as f:
        vocab = pickle.load(f)

    # Build models
    encoder = EncoderCNN(args.embed_size).eval()  # eval mode (batchnorm uses moving mean/variance)
    decoder = DecoderRNN(args.embed_size, args.hidden_size, len(vocab), args.num_layers)

    dissection.replace_layers(encoder, [
        ('resnet.7.2.bn3', 'final_layer'),
    ])
    vec = torch.zeros(2048).to(device)
    vec[0] = 100
    encoder.replacement['final_layer'] = vec

    encoder = encoder.to(device)
    decoder = decoder.to(device)

    # Load the trained model parameters
    encoder.load_state_dict(torch.load(args.encoder_path))
    decoder.load_state_dict(torch.load(args.decoder_path))

    # Prepare an image
    image = load_image(args.image, transform)
    image_tensor = image.to(device)
    
    # Generate an caption from the image
    feature = encoder(image_tensor)
    sampled_ids = decoder.sample(feature)
    sampled_ids = sampled_ids[0].cpu().numpy()          # (1, max_seq_length) -> (max_seq_length)
    
    # Convert word_ids to words
    sampled_caption = []
    for word_id in sampled_ids:
        word = vocab.idx2word[word_id]
        sampled_caption.append(word)
        if word == '<end>':
            break
    sentence = ' '.join(sampled_caption)
    
    # Print out the image and the generated caption
    print (sentence)
    # image = Image.open(args.image)
    # plt.imshow(np.asarray(image))


def main(args):
    sample(args)
    # get_covariance(args)

    
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
