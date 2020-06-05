# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 18:49:56 2020

@author: kosprpv69
"""
import numpy as np
from collections import OrderedDict,defaultdict



def calc_distance(detections,height):
      """ detections is a list of all detections found in a frame
      of the form (object,confidence,(x,y,w,h)). This function returns the 
      detections which have violated and ones which havent"""
      
      ranges = [i for i in range(int(height/8),height+1,int(height/8))]

      slots = OrderedDict({i:[] for i in ranges})

      for i in detections:
            for j in ranges:
                  if i[2][1]<j:
                        slots[j].append(i)
                        break
      
      detections_updated = []
      
      for i,j in slots.items():
            if j:
#                  j = sorted(j,key = lambda x: x[2][3])
                  heights = [k[2][3] for k in j]
                  height_median = np.median(heights)
                  new_j = [(height_median,k[1],k[2]) for k in j]
                  detections_updated.extend(new_j)
     
      detections_updated.sort(key = lambda x: x[2][0])
      
      violations_set = set()
      for j,i in enumerate(detections_updated):
            for k in detections_updated[j+1:]:
                  if abs(i[2][0]-k[2][0])<i[0]: 
                        dist =  np.sqrt((i[2][0]-k[2][0])**2 + (2*(i[2][1]-k[2][1]))**2)
                        if dist < i[0] and dist < k[0]:
                              violations_set.add(i)
                              violations_set.add(k)
                  else:
                        break
      non_violations_set = set(detections_updated).difference(violations_set)
      return list(violations_set),list(non_violations_set)
      

#calc_distance([(b'person', 0.9359182119369507, (65.48546600341797, 377.8138122558594, 13.299861907958984, 68.51302337646484)), (b'person', 0.7065868973731995, (364.851806640625, 362.12603759765625, 10.6348876953125, 44.321414947509766)), (b'person', 0.6553335785865784, (367.4599304199219, 302.4787292480469, 7.2274322509765625, 27.225645065307617)), (b'person', 0.6338825225830078, (98.61302185058594, 378.11175537109375, 10.23060131072998, 60.04832458496094)), (b'person', 0.5679501295089722, (268.466552734375, 316.7518615722656, 9.496596336364746, 33.531333923339844)), (b'person', 0.5622066259384155, (342.1731262207031, 429.3646240234375, 17.699085235595703, 56.791690826416016)), (b'person', 0.4988653361797333, (253.48260498046875, 282.7754821777344, 5.330770015716553, 24.58672523498535)), (b'person', 0.4853569567203522, (339.87677001953125, 347.426513671875, 10.744488716125488, 22.269214630126953)), (b'person', 0.4652334749698639, (109.78838348388672, 358.4297180175781, 9.001030921936035, 59.537471771240234)), (b'person', 0.34086450934410095, (269.6197204589844, 282.42388916015625, 6.279255390167236, 22.353992462158203)), (b'person', 0.30431637167930603, (281.0084228515625, 270.2095947265625, 4.727065086364746, 19.621034622192383))],608)