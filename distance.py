# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 18:49:56 2020

@author: kosprpv69
"""
import numpy as np
from collections import OrderedDict,defaultdict
from bisect import bisect_left,bisect_right


def approx_area(to_check,left,right):
      dist_left = to_check[2][1]-left[2][1]
      dist_right = right[2][1]-to_check[2][1]
      
      sqrt_area_left = np.sqrt(left[2][2]*left[2][3])
      sqrt_area_right = np.sqrt(right[2][2]*right[2][3])
      
      if dist_left+dist_right>0:
            approx = (dist_left*sqrt_area_right + dist_right*sqrt_area_left)/(dist_left+dist_right)
      else:
            approx = (sqrt_area_right + sqrt_area_left)/2
      return approx
      

def validate_area(detections,dims):
      """Calculates approx height based on h values and y coordinates"""
      invalid = set()
      detections = list(filter(lambda x: x[2][2]*x[2][3] < 0.2*dims[0]*dims[1],detections))
      
      if len(detections)>=3:
            detections.sort(key= lambda x: x[2][1])
            for num,detection in enumerate(detections):
                  if (num-1) >= 0 and (num+1) <= len(detections)-1 :
                        area = approx_area(detection,detections[num-1],detections[num+1])
                  elif (num-1) < 0:
                        area = approx_area(detection,detections[num+1],detections[num+2])
                  else:
                        area = approx_area(detection,detections[num-2],detections[num-1])
                  
                  if np.sqrt(detection[2][3]*detection[2][2])/area > 2:
                        invalid.add(detection)
      
      return list(set(detections).difference(invalid))



def combine(persons,motorbikes):
      """ This function combines the detection of persons and motorbike to get
      greater number of predictions and more accurate readings of the height of
      person riding a motorbike. Returns combined detection arrays of """

      motorbikes = sorted(motorbikes, key = lambda x: x[2][0])
      to_drop = set()
      if persons:
            persons = sorted(persons,key = lambda x: x[2][0])
            persons_x = [x[2][0] for x in persons]
            persons_y = [x[2][1] for x in persons]

            for i in motorbikes:
                  left_ind = bisect_left(persons_x, i[2][0]-i[2][2]/4)
                  right_ind = bisect_right(persons_x,i[2][0]+i[2][2]/4)
                  if left_ind != right_ind :
                        dists = [(num,j-i[2][1]) for num,j in             
                                 enumerate(persons_y[left_ind:right_ind],start= left_ind)
                                 if j>(i[2][1]-1*i[2][3]) and j<i[2][1]]
                        if dists:
                              dists = max(dists,key= lambda x:x[1])
                              to_drop.add(i)
                              temp = persons[dists[0]]
                              new_max_y = max(i[2][1]+i[2][3]/2,temp[2][1]+temp[2][3]/2)
                              new_h = new_max_y - (temp[2][1]-temp[2][3]/2)
                              new_y = (temp[2][1]-temp[2][3]/2) + new_h/2
                              persons[dists[0]] = (temp[0],temp[1],(temp[2][0],new_y,temp[2][2],new_h))
                              #print(temp,persons[dists[0]])
                              
      motorbikes = list(set(motorbikes).difference(to_drop))
      
      to_drop = set()
      for motorbike in motorbikes:
            if motorbike[2][3]/motorbike[2][2] >= 2:
                  persons.append(motorbike)
                  to_drop.add(motorbike)
      
      motorbikes = list(set(motorbikes).difference(to_drop))
      
      return motorbikes,persons
                              

def calc_distance(detections,height):
      """ 
      Function to calculate the detections found to be in violation of 
      social distancing
      
      Parameters
      ----------------------------------------------------
      detections: iter
            List of detections containing object name, confidence level
            and (x_cord,y_cord,width,height) of the bounding box
      
      height: integer
            Total height in pixels of the frame
            
      Returns
      ----------------------------------------------------
      violations: iter
            List of detections which were found to be violating the distance
            measure
      
      Non-violations: iter
            List of detections which were not violating the distance measure
      """
      
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
      
detections = [(b'person', 0.9287652373313904, (117.55171966552734, 357.0683288574219, 10.45461654663086, 57.01490020751953)), (b'person', 0.904653787612915, (71.09352111816406, 372.96807861328125, 11.777061462402344, 58.30332565307617)), (b'person', 0.8984009027481079, (315.1519775390625, 489.4537353515625, 22.12396812438965, 73.49960327148438)), (b'person', 0.8768723607063293, (367.31524658203125, 467.0140686035156, 17.59693145751953, 83.31715393066406)), (b'motorbike', 0.8234314918518066, (330.6274719238281, 395.3494873046875, 9.748992919921875, 35.824947357177734)), (b'motorbike', 0.8130174875259399, (314.6607971191406, 514.8377685546875, 21.29739761352539, 60.6729736328125)), (b'person', 0.7671158909797668, (84.20451354980469, 380.73736572265625, 16.01509666442871, 61.79273986816406)), (b'motorbike', 0.7440154552459717, (324.5339050292969, 308.2414855957031, 8.691008567810059, 24.17902374267578)), (b'person', 0.712255597114563, (174.29046630859375, 374.1143493652344, 18.199432373046875, 50.20158386230469)), (b'motorbike', 0.7111533880233765, (172.19300842285156, 388.95257568359375, 16.135021209716797, 38.54924392700195)), (b'person', 0.705040454864502, (353.13519287109375, 382.6665344238281, 12.666882514953613, 48.527442932128906)), (b'motorbike', 0.6527384519577026, (351.4810485839844, 397.3664245605469, 10.762608528137207, 33.73689651489258)), (b'person', 0.6058199405670166, (245.77651977539062, 294.17724609375, 8.104937553405762, 26.522705078125)), (b'motorbike', 0.5371377468109131, (298.3529968261719, 274.3265380859375, 6.9689178466796875, 15.561272621154785)), (b'person', 0.5362447500228882, (330.9190673828125, 379.0393371582031, 12.697729110717773, 51.01247024536133)), (b'motorbike', 0.532401978969574, (281.2871398925781, 275.159423828125, 6.368981838226318, 23.13214683532715)), (b'motorbike', 0.5307982563972473, (367.574951171875, 307.8184814453125, 6.746595859527588, 28.050201416015625)), (b'motorbike', 0.45152127742767334, (245.5825958251953, 297.759033203125, 7.620977401733398, 22.562793731689453))]

persons  = [i for i in detections if i[0].decode('ASCII')  == 'person']
motorbikes  = [i for i in detections if i[0].decode('ASCII')  == 'motorbike']

combine(persons,motorbikes)