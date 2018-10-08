import os, torch, numpy, csv
import skimage.morphology
from collections import OrderedDict
from netdissect.segmodel import ModelBuilder, SegmentationModule

class GanImageSegmenter:
    def __init__(self, segarch=None, segvocab=None, segsizes=None,
            multilabel=0, segdiv=None, epoch=None, segnormed=False):
        # Create a segmentation model
        if segvocab == None:
            segvocab = 'baseline'
        if segarch == None:
            segarch = ('resnet50_dilated8', 'ppm_bilinear_deepsup')
        if segdiv == None:
            segdiv = 'undivided'
        if epoch is None:
            epoch = 20
        elif isinstance(segarch, str):
            segarch = segarch.split(',')
        if segsizes is None:
            segsizes = [256, 384, 480, 576]
        segmodel = load_segmentation_model(segarch, segvocab, epoch)
        segmodel.cuda()
        self.segmodel = segmodel
        self.multilabel = multilabel
        self.segdiv = segdiv
        self.segnormed = segnormed
        mult = 1
        if self.segdiv == 'quad':
            mult = 5
        elif self.segdiv == 'oct':
            mult = 9
        self.segsizes = segsizes
        # Examine segmentation model to determine number of seg classes
        self.num_underlying_classes = (list(c for c in segmodel.modules()
            if isinstance(c, torch.nn.Conv2d)))[-1].out_channels
        self.num_classes = self.num_underlying_classes * mult + 1
        # (note add one to reserve zero to mean 'no label'
        # Current source code directory, for finding csv data
        srcdir = os.path.realpath(
           os.path.join(os.getcwd(), os.path.dirname(__file__)))
        # Load csv of class names
        assert len(self.segmodel.labels) * mult + 1 == self.num_classes

    def get_label_and_category_names(self, dataset):
        labelcats = [(label, self.segmodel.categories[c]) for label, c in
                zip(self.segmodel.labels, self.segmodel.label_category)]
        suffixes = []
        if self.segdiv == 'quad':
            suffixes = ['t', 'l', 'b', 'r']
        elif self.segdiv == 'oct':
            suffixes = ['t', 'l', 'b', 'r',
                    'tl', 'tr', 'bl', 'br']
        divided_labels = []
        for suffix in suffixes:
            divided_labels.extend([('%s-%s' % (label, suffix), cat)
                for label, cat in labelcats])
        labelcats.extend(divided_labels)
        # Offset labels by one
        return [('-', self.segmodel.categories[0])
                ] + labelcats, self.segmodel.categories

    def reverse_normalize(self, image):
        return (image + 1) / 2

    def recover_image_and_features(self, batch, model):
        device = next(model.parameters()).device
        z_batch = batch[0]
        byte_im = (self.reverse_normalize(model(z_batch.to(device))) * 255)
        byte_im = byte_im.permute(0, 2, 3, 1).clamp(0, 255).byte()
        return byte_im, getattr(model, 'retained', None), None

    def recover_im_seg_bc_and_features(self, batch, model, byte_images=False):
        device = next(model.parameters()).device
        z_batch = batch[0]
        tensor_images = model(z_batch.to(device))
        seg = self.segment_batch(tensor_images, downsample=2)
        index = torch.arange(z_batch.shape[0], dtype=torch.long, device=device)
        bc = (seg + index[:, None, None, None] * self.num_classes).view(-1
            ).bincount(minlength=z_batch.shape[0] * self.num_classes)
        bc = bc.view(z_batch.shape[0], self.num_classes)
        # Note images are not byteified
        if byte_images:
            images = (self.reverse_normalize(tensor_images) * 255)
            images = images.permute(0, 2, 3, 1).clamp(0, 255).byte()
        else:
            images = tensor_images
        return images, seg, bc, getattr(model, 'retained', None), None

    def segment_batch(self, tensor_images, downsample=1, multilabel=0):
        '''
        Generates a segmentation by applying multiresolution voting on
        the segmentation model, using (rounded to 32 pixels) a set of
        resolutions in the example benchmark code.
        '''
        y, x = tensor_images.shape[2:]
        b = len(tensor_images)
        if not self.segnormed:
           tensor_images = (tensor_images + 1) / 2 * 255
        else:
            # Maybe this is better?
            tensor_images = fwdnorm((tensor_images + 1) / 2) # * 255
        seg_shape = (y // downsample, x // downsample)
        # We want these to be multiples of 32 for the model.
        sizes = [(s, s) for s in self.segsizes]
        pred = torch.zeros(
            len(tensor_images), (self.num_underlying_classes),
            seg_shape[0], seg_shape[1]).cuda()
        for size in sizes:
            if size == tensor_images.shape[2:]:
                resized = tensor_images
            else:
                resized = torch.nn.AdaptiveAvgPool2d(size)(tensor_images)
                # The following smarter resize code doesn't seem to work yet.
                # grid1 = torch.linspace(-1, 1, size[0])
                # grid2 = torch.linspace(-1, 1, size[1])
                # grid = torch.cat((
                #     grid1[None,:,None,None].expand(1, -1, len(grid2), -1),
                #     grid2[None,None,:,None].expand(1, len(grid1), -1, -1)),
                #              dim=3).expand(b, -1, -1, -1).to(tensor_images)
                # resized = torch.nn.functional.grid_sample(tensor_images, grid)
            r_pred = self.segmodel(
                dict(img_data=resized), segSize=seg_shape)
            pred += r_pred
        if multilabel == 0:
            _, segs = torch.max(pred, dim=1)
            segs += 1
            segs = segs[:, None, :, :]
        else:
            vals, segs = torch.topk(pred, multilabel, dim=1)
            segs += 1
            segs[vals <= 0] = 0
        if self.segdiv == 'quad' or self.segdiv == 'oct':
            segs = self.expand_segment_quad(segs,
                    self.num_underlying_classes, self.segdiv)
        return segs

    def expand_segment_quad(self, segs, num_seg_labels, segdiv='quad'):
        shape = segs.shape
        output = segs.repeat(1, (4 if segdiv == 'oct' else 3), 1, 1)
        # For every connected component present (using generator)
        for i, mask in component_masks(segs):
            # Figure the bounding box of the label
            top, bottom = mask.any(dim=1).nonzero()[[0, -1], 0]
            left, right = mask.any(dim=0).nonzero()[[0, -1], 0]
            # Chop the bounding box into four parts
            vmid = (top + bottom + 1) // 2
            hmid = (left + right + 1) // 2
            # Construct top, bottom, right, left masks
            quad_mask = mask[None,:,:].repeat(4, 1, 1)
            quad_mask[0, vmid:, :] = 0   # top
            quad_mask[1, :, hmid:] = 0   # right
            quad_mask[2, :vmid, :] = 0   # bottom
            quad_mask[3, :, :hmid] = 0   # left
            quad_mask = quad_mask.long()
            # Modify extra segmentation labels by offsetting
            output[i,1,:,:] += quad_mask[0] * num_seg_labels
            output[i,2,:,:] += quad_mask[1] * (2 * num_seg_labels)
            output[i,1,:,:] += quad_mask[2] * (3 * num_seg_labels)
            output[i,2,:,:] += quad_mask[3] * (4 * num_seg_labels)
            if segdiv == 'oct':
                output[i,3,:,:] += (
                        quad_mask[0] * quad_mask[1] * (5 * num_seg_labels))
                output[i,3,:,:] += (
                        quad_mask[0] * quad_mask[3] * (6 * num_seg_labels))
                output[i,3,:,:] += (
                        quad_mask[2] * quad_mask[1] * (7 * num_seg_labels))
                output[i,3,:,:] += (
                        quad_mask[2] * quad_mask[3] * (8 * num_seg_labels))
        return output

def label_masks(segmentation_batch):
    '''
    Treats entire label as one region.
    '''
    for i in range(segmentation_batch.shape[0]):
        for label in torch.bincount(segmentation_batch[i].flatten()).nonzero():
            if label == 0:
                continue
            yield i, (segmentation_batch[i] == label)[0]

def component_masks(segmentation_batch):
    '''
    Splits connected components into regions (slower, requires cpu).
    '''
    npbatch = segmentation_batch.cpu().numpy()
    for i in range(segmentation_batch.shape[0]):
        labeled, num = skimage.morphology.label(npbatch[i][0], return_num=True)
        labeled = torch.from_numpy(labeled).to(segmentation_batch.device)
        for label in range(1, num):
            yield i, (labeled == label)


def standard_z_sample(size=100, depth=1024, seed=1, device=None):
	'''
	Generate a standard set of random Z as a (size, z_dimension) tensor.
	With the same random seed, it always returns the same z (e.g.,
	the first one is always the same regardless of the size.)
	'''
	# Use numpy RandomState since it can be done deterministically
	# without affecting global state
	rng = numpy.random.RandomState(seed)
	result = torch.from_numpy(
			rng.standard_normal(size * depth)
			.reshape(size, depth)).float()
	if device is not None:
		result = result.to(device)
	return result

def load_segmentation_model(segmodel_arch, segvocab, epoch=20):
    # Load csv of class names
    segmodel_dir = 'dataset/segmodel/%s-%s-%s' % ((segvocab,) + segmodel_arch)
    labeldata = list(read_csv(os.path.join(segmodel_dir, 'labels.csv')))
    # Create a segmentation model
    segbuilder = ModelBuilder()
    # example segmodel_arch = ('resnet101', 'upernet')
    seg_encoder = segbuilder.build_encoder(
            arch=segmodel_arch[0],
            fc_dim=2048,
            weights=os.path.join(segmodel_dir, 'encoder_epoch_%d.pth' % epoch))
    seg_decoder = segbuilder.build_decoder(
            arch=segmodel_arch[1],
            fc_dim=2048, use_softmax=True, num_class=len(labeldata),
            weights=os.path.join(segmodel_dir, 'decoder_epoch_%d.pth' % epoch))
    segmodel = SegmentationModule(seg_encoder, seg_decoder,
                                  torch.nn.NLLLoss(ignore_index=-1))
    labeldata = list(read_csv(os.path.join(segmodel_dir, 'labels.csv')))
    segmodel.labels = [r['Name'].split(';', 1)[0] for r in labeldata]
    categories = OrderedDict()
    label_category = numpy.zeros(len(segmodel.labels), dtype=int)
    if 'Category' in labeldata[0]:
        for i, r in enumerate(labeldata):
            cat = r['Category']
            if cat not in categories:
                categories[cat] = len(categories)
            label_category[i] = categories[cat]
        segmodel.categories = list(categories.keys())
    else:
        segmodel.categories = ['object']
    segmodel.label_category = label_category
    segmodel.eval()
    return segmodel

def read_csv(filename):
    with open(filename, 'r') as f:
        return list(csv.DictReader(f))

IMAGE_MEAN = [0.485, 0.456, 0.406]
IMAGE_STDEV = [0.229, 0.224, 0.225]
T_IMAGE_MEAN = torch.from_numpy(numpy.array(IMAGE_MEAN)).float()
T_IMAGE_STDEV = torch.from_numpy(numpy.array(IMAGE_STDEV)).float()

def fwdnorm(imgbatch):
    return imgbatch.sub(
            T_IMAGE_MEAN[:,None,None].to(imgbatch.device)
     ).div_(T_IMAGE_STDEV[:,None,None].to(imgbatch.device))

if __name__ == '__main__':
    main()
