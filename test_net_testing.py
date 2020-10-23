from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import pandas as pd

import _init_paths
import os
import sys
import numpy as np
import argparse
import pprint
import pdb
import time
import cv2
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import pickle
from roi_data_layer.roidb import combined_roidb
from roi_data_layer.roibatchLoader import roibatchLoader
from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from model.rpn.bbox_transform import clip_boxes
# from model.nms.nms_wrapper import nms
from model.roi_layers import nms
from model.rpn.bbox_transform import bbox_transform_inv
from model.utils.net_utils import save_net, load_net, vis_detections
from model.faster_rcnn.vgg16 import vgg16
from model.faster_rcnn.resnet import resnet

import pdb

import matplotlib.image as mpimg
import matplotlib.pyplot as plt



try:
    xrange          # Python 2
except NameError:
    xrange = range  # Python 3


lr = cfg.TRAIN.LEARNING_RATE
momentum = cfg.TRAIN.MOMENTUM
weight_decay = cfg.TRAIN.WEIGHT_DECAY

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', dest='cuda', help='whether use CUDA', action='store_true')
    parser.add_argument('--models',default='res50')
    parser.add_argument('--dataset', default='coco')
    parser.add_argument('--class_agnostic', default=True)
    parser.add_argument('--load_dir', dest='load_dir', help='directory to load models', default="models", type=str)
    parser.add_argument('--s', dest='checksession', help='checksession to load model', default=1, type=int)
    parser.add_argument('--checkepoch', dest='checkepoch', help='checkepoch to load network', default=10, type=int)
    parser.add_argument('--p', dest='checkpoint', help='checkpoint to load network', default=1663, type=int)
    parser.add_argument('--vis', dest='vis', help='visualization mode', action='store_true')
    parser.add_argument('--a', dest='average', help='average the top_k candidate samples', default=1, type=int)


    args = parser.parse_args()

    print('Called with args:')
    print(args)

    if torch.cuda.is_available() and not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    
    np.random.seed(cfg.RNG_SEED)

    args.imdb_name = "coco_2017_train"
    args.imdbval_name = "coco_2017_val"
    args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']

    #Grupo y net a Utilizar
    args.group= 0
    args.net = 'res50'

    args.cfg_file = "cfgs/{}_{}.yml".format(args.net, args.group) if args.group != 0 else "cfgs/{}.yml".format(args.net)

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    print('\nUsando configuracion:')
    pprint.pprint(cfg)

    args.seen=True
    print("\nSe carga el dataset:")
    cfg.TRAIN.USE_FLIPPED = False
    imdb_vu, roidb_vu, ratio_list_vu, ratio_index_vu, query_vu = combined_roidb(args.imdbval_name, False, seen=args.seen)
    print('Clases existentes en ell modelo  (largo :', len(imdb_vu.classes), ')')
    print('imdb_vu.classes', imdb_vu.classes)

    print('ratio_list_vu', ratio_list_vu.shape)
    
    print('ratio_index_vu', ratio_index_vu.shape)
    
    print('query_vu', len(query_vu))

    imdb_vu.competition_mode(on=True)
    

    dataset_vu = roibatchLoader(roidb_vu, ratio_list_vu, ratio_index_vu, query_vu, 1, imdb_vu.num_classes, training=False, seen=args.seen)

    # Inicializar la red

    print("\nInicialización de la red")
    if args.net == 'vgg16':
      fasterRCNN = vgg16(imdb_vu.classes, pretrained=False, class_agnostic=args.class_agnostic)
    elif args.net == 'res101':
      fasterRCNN = resnet(imdb_vu.classes, 101, pretrained=False, class_agnostic=args.class_agnostic)
    elif args.net == 'res50':
      fasterRCNN = resnet(imdb_vu.classes, 50, pretrained=False, class_agnostic=args.class_agnostic)
    elif args.net == 'res152':
      fasterRCNN = resnet(imdb_vu.classes, 152, pretrained=False, class_agnostic=args.class_agnostic)
    else:
      print("network is not defined")
      pdb.set_trace()
    fasterRCNN.create_architecture()

    # Load checkpoint model
    input_dir = args.load_dir + "/" + args.net + "/" + args.dataset
    print('\nUtilizando modelo:', input_dir)
    if not os.path.exists(input_dir):
      raise Exception('There is no input directory for loading network from ' + input_dir)
    
    #Cargar el modelo en la arquitectura cargada en la sesion 1, epoca 1 
    load_name = os.path.join(input_dir,
    'faster_rcnn_{}_{}_{}.pth'.format(args.checksession, args.checkepoch, args.checkpoint))
    print("load checkpoint %s" % (load_name))
    checkpoint = torch.load(load_name)
    print("checkpoint models keys:", checkpoint.keys())


    fasterRCNN.load_state_dict(checkpoint['model'])

    print("Red con pesos ya cargados")


    print("\nInicialización de Holders")
    # initilize the tensor holder here.
    im_data = torch.FloatTensor(1)
    query   = torch.FloatTensor(1)
    im_info = torch.FloatTensor(1)
    catgory = torch.LongTensor(1)
    gt_boxes = torch.FloatTensor(1)

    print('\nPaso de tansores a cuda.')
    # ship to cuda
    if args.cuda:
      cfg.CUDA = True
      fasterRCNN.cuda()
      im_data = im_data.cuda()
      query = query.cuda()
      im_info = im_info.cuda()
      catgory = catgory.cuda()
      gt_boxes = gt_boxes.cuda()

    print("\nPasar los holders a Variable (Conversion de nodos de pytorch, hoy en día esto estadeprecado.)")
    # make variable
    im_data = Variable(im_data)
    query = Variable(query)
    im_info = Variable(im_info)
    catgory = Variable(catgory)
    gt_boxes = Variable(gt_boxes)


    # record time
    start = time.time()

    # visiualization
    vis = args.vis
    if vis:
      thresh = 0.05
    else:
      thresh = 0.0
    max_per_image = 100

    print('\nCreacion de directorio de salida')
    # create output Directory
    output_dir_vu = get_output_dir(imdb_vu, 'faster_rcnn_unseen')

    print('\nSetear la red para evaluacion')
    fasterRCNN.eval()

    for avg in range(args.average):

      dataset_vu.query_position = avg
      print("\nCreacion del dataloader")
      dataloader_vu = torch.utils.data.DataLoader(dataset_vu, batch_size=1,shuffle=False, num_workers=0,pin_memory=True)

      print("\nCreacion de iterador del dataloader")
      data_iter_vu = iter(dataloader_vu)

      
      # total quantity of testing images, each images include multiple detect class
      num_images_vu = len(imdb_vu.image_index)
      print("num_images_vu: Cantidad total de imagenes de testeo, cada una incluye multiples clases de deteccion:", num_images_vu)
      num_detect = len(ratio_index_vu[0])
      print('num_detect:', num_detect )

      #Recopilacion de todas las clases 
      all_boxes = [[[] for _ in xrange(num_images_vu)]
                  for _ in xrange(imdb_vu.num_classes)]
      #print("all_boxes",all_boxes[all_boxes==True])
      

      
      _t = {'im_detect': time.time(), 'misc': time.time()}
      if args.group != 0:
        det_file = os.path.join(output_dir_vu, 'sess%d_g%d_seen%d_%d.pkl'%(args.checksession, args.group, args.seen, avg))
      else:
        det_file = os.path.join(output_dir_vu, 'sess%d_seen%d_%d.pkl'%(args.checksession, args.seen, avg))
      print('det_file', det_file)

      if os.path.exists(det_file):
        with open(det_file, 'rb') as fid:
          all_boxes = pickle.load(fid)
      else:
        for i,index in enumerate(ratio_index_vu[0]):
          
          data = next(data_iter_vu)
          

          im_data.data.resize_(data[0].size()).copy_(data[0])
          query.data.resize_(data[1].size()).copy_(data[1])
          im_info.data.resize_(data[2].size()).copy_(data[2])
          gt_boxes.data.resize_(data[3].size()).copy_(data[3])
          catgory.data.resize_(data[4].size()).copy_(data[4])

          plt.figure()
          plt.imshow(im_data[0].permute(1,2,0))

          plt.figure()
          plt.imshow(query[0].permute(1,2,0))
          plt.show()

          # Run Testing
          print('\nEntradas al modelo:')
          print('im_data: Data de la imágen', im_data.shape)
          print('query:', query.shape)
          print('im_info', im_info)
          print('gt_boxes', gt_boxes)

          

          det_tic = time.time()
          rois, cls_prob, bbox_pred, \
          rpn_loss_cls, rpn_loss_box, \
          RCNN_loss_cls, _, RCNN_loss_bbox, \
          rois_label, weight = fasterRCNN(im_data, query, im_info, gt_boxes, catgory)


          scores = cls_prob.data
          boxes = rois.data[:, :, 1:5]


