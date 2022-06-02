import torch
from torch.utils.data import DataLoader
from torch.utils import data
from torchvision import datasets
from torchvision import transforms as T
import torchvision.transforms.functional as TF
import os
import random
import cv2
import numpy as np
import segyio

from aux_functions_demo import *


class SeismicDataset(data.Dataset):
    def __init__(self, root, dataset_name, split,train_type='sup_ssl', task=None, n_few_shot=None,cross_val=False, fold=None,present_sec='inline'):
        self.root = root       
        self.split = split.lower()
        self.dataset_name =  dataset_name
        self.train_type = train_type
        self.partition = [0.60,0.40,0.00]  # Originally: [0.60,0.20,0.20]

        self.task = task
        self.rotations = [-8,-4,0,4,8]
        self.fold = fold

        if self.train_type == 'few_shot' or self.train_type == 'fine_tune':
            assert n_few_shot != None, 'None Type is not valid'
            self.n_few_shot = n_few_shot 
        
        
        assert self.dataset_name=='F3_netherlands' or self.dataset_name=='Parihaka_NZPM',f"Not recognized dataset {self.dataset_name}"
        assert self.split=='train' or self.split=='val' or self.split=='test' , f"Not recognized split type {self.split}"
        assert self.task== None or self.task=='segmentation' or self.task=='rotation' or self.task=='jigsaw', f"Not recognized task{self.task}"
        assert self.train_type=='sup_ssl' or self.train_type=='few_shot'or self.train_type=='fine_tune', f"Not recognized train type type {self.train_type}"

        if self.task =='jigsaw':
            self.possible_permutations = generate_random_permutations()

        # Creating list of image files.
        if self.dataset_name=='F3_netherlands':
            
            self.facies = {0:"See Bottom/Rock Interaction",1:"North Sea Supergroup",2: "Chalk Group", 3:"Rijnland Group",4: "Schieland, Scruff and Niedersachsen Group",
                        5: "Altena Group" ,6: "Germanic Trias Group", 7:"Zechstein Group" ,8: "Rotliegend Group",9: "Carboniferous Group"}
            self.classes = np.arange(0,len(self.facies.keys()))
            self.sections, self.masks_list, self.sec_number_list = self.make_dataset()


        elif self.dataset_name=='Parihaka_NZPM':
            self.sections, self.sec_number_list = self.make_dataset()

            self.facies = {0:"Basement/Other: Basement",1:"Slope Mudstone A", 2:"Mass Transport Deposit",
                                    3: "Slope Mudstone B",4: "Slope Valley" ,5: "Submarine Canyon System"}
            self.classes = np.arange(0,len(self.facies.keys()))

        else:
            raise ValueError(f'Not recognized dataset {self.dataset_name}')

            
    def make_dataset(self):
        
        # Specify Paths
        if self.dataset_name=='F3_netherlands':
            
            self.masks_path = 'masks/'
            self.data_path = 'sections/'

            # Joining input paths.
            self.data_path = os.path.join(self.root, self.dataset_name,
                                        self.data_path)
            self.masks_path = os.path.join(self.root, self.dataset_name,
                                        self.masks_path)
            
            #Read all sections in folder
            sections_list = []
            masks_list = []
            sec_number_list = []

            files = os.listdir(self.data_path)
            filtered_files = sorted([ r for r in files if  r.split('.')[-1]=='tiff' ])
                
            for filename in filtered_files:
                sections_list.append(os.path.join(self.data_path,filename))
                sec_number = filename.split("/")[-1].split(".")[0]
                masks_list.append(os.path.join(self.masks_path)+str(sec_number)+'_mask.png')
                sec_number_list.append(sec_number)

            seed = 42 
            np.random.seed(seed) ; sections_list = list(np.random.permutation(sections_list))
            np.random.seed(seed) ; masks_list = list(np.random.permutation(masks_list))
            np.random.seed(seed) ; sec_number_list = list(np.random.permutation(sec_number_list))


            if self.split=='train':
                high = int(np.round((len(sections_list) * self.partition[0]),0)) #+ 1 
                sections_list = sections_list[:high]
                masks_list = masks_list[:high]
                sec_number_list = sec_number_list[:high]
            elif self.split=='test':   
                low = int(np.round((len(sections_list) * self.partition[0]),0)) #+1
                high = int(np.round((len(sections_list) * self.partition[1]),0))
                sections_list = sections_list[low:-high]
                masks_list = masks_list[low:-high]
                sec_number_list = sec_number_list[low:-high]
            elif self.split=='val':
                low = int(np.round((len(sections_list) * self.partition[2]),0)) # +1
                sections_list = sections_list[-low:]
                masks_list = masks_list[-low:]
                sec_number_list = sec_number_list[-low:]

            else:
                raise ValueError(f'Not recognized split type {self.split}')

            #Keeping only N train sections can be maintaned
            if self.split=='train' and (self.train_type == 'few_shot' or self.train_type == 'fine_tune'):
                sec_few = []
                mask_few = []
                sec_number_few = []

                #sec_index = np.random.randint(0,len(sections_list),size=self.n_few_shot) 
                sec_few = sections_list[:self.n_few_shot]
                mask_few = masks_list[:self.n_few_shot]
                sec_number_few = sec_number_list[:self.n_few_shot]

                return sec_few, mask_few,sec_number_few

            else:
                return sections_list, masks_list, sec_number_list

        #### New Zealand Parihaka  #####
        #######################################################################################################
        elif self.dataset_name=='Parihaka_NZPM':
            raise ValueError('Parihaka dataset is not yet available in this demo')

            '''

            """
            As we do not have labels for the test image, let us split the train set into three sets:
            train, val and test 
            """
            
            self.masks_path = 'TrainingData_Labels.segy'
            self.data_path = 'TrainingData_Image.segy'

            # Joining input paths.
            self.data_path = os.path.join(self.root, self.dataset_name,
                                        self.data_path)
            self.masks_path = os.path.join(self.root, self.dataset_name,
                                           self.masks_path) 

            self.section_volume = segyio.tools.cube(self.data_path)
            if self.task == None or self.task=='segmentation':
                self.label_volume = segyio.tools.cube(self.masks_path)
                for i in range(0,6):
                    self.label_volume = np.where(self.label_volume ==i+1,i,self.label_volume)

                assert self.label_volume.shape == self.section_volume.shape, f'Expected same shape for section and label, but got: {self.label_volume.shape} and {self.section_volume.shape}'

            self.section_volume =  np.transpose(self.section_volume, (2, 0, 1))
            
            if self.task == None or self.task=='segmentation':
                self.label_volume =  np.transpose(self.label_volume, (2, 0, 1))
                assert self.label_volume.shape == self.section_volume.shape, f'Expected same shape for section and label (1006,590,782), but got: {self.label_volume.shape} and {self.section_volume.shape}'

            inline = np.arange(0,self.section_volume.shape[1])
            crossline = np.arange(0,self.section_volume.shape[2])

            seed = 42 #np.random.randint(3125161651,size=1)
            np.random.seed(seed) ; inline = list(np.random.permutation(inline))
            np.random.seed(seed) ; crossline = list(np.random.permutation(crossline))

            if self.split=='train':

                if self.train_type == 'sup_ssl':
                    high_i =  int(np.round((self.section_volume.shape[1] * self.partition[0]),0))
                    high_c =  int(np.round((self.section_volume.shape[2] * self.partition[0]),0))
                    inline = inline[:high_i]
                    crossline = crossline[:high_c]

                elif self.train_type == 'few_shot' or self.train_type == 'fine_tune':
                    if self.n_few_shot == 1:
                        inline = inline[:1]
                        inline_name = ['Inline_' + z for z in [str(i) for i in inline]]
                        return inline, inline_name
                    else:
                        top_up = int(self.n_few_shot/2)
                        inline = inline[:top_up]
                        if self.n_few_shot % 2 != 0:
                            top_up = top_up+1
                        crossline = crossline[:top_up]

                else:
                    raise ValueError(f'Not recognized train type: {self.train_type}')

            elif self.split=='val':  
                low_i =  int(np.round((self.section_volume.shape[1] * self.partition[0]),0))
                low_c =  int(np.round((self.section_volume.shape[2] * self.partition[0]),0))

                high_i =  int(np.round((self.section_volume.shape[1] * self.partition[1]),0))
                high_c =  int(np.round((self.section_volume.shape[2] * self.partition[1]),0))

                inline = inline[low_i:-(high_i)]
                crossline = crossline[low_c:-high_c]

            elif self.split=='test':
                low_i =  int(np.round((self.section_volume.shape[1] * self.partition[2]),0))
                low_c =  int(np.round((self.section_volume.shape[2] * self.partition[2]),0))

                inline = inline[-(low_i):]
                crossline = crossline[-low_c:]


            else:
                raise ValueError(f'Not recognized split type {self.split}')
            


            inline_name = ['Inline_' + z for z in [str(i) for i in inline]]
            crossline_name = ['Crossline_' + z for z in [str(i) for i in crossline]]
            sections = np.hstack(np.array([inline, crossline]))
            sec_name = inline_name + crossline_name

            return sections,sec_name '''
        
        else:
            raise ValueError(f'Not recognized dataset {self.dataset_name}')
        
        
    def __getitem__(self, index):

        if self.dataset_name=='Parihaka_NZPM':

            raise ValueError('Parihaka dataset is not yet available in this demo')

            '''

            idx = self.sections[index]
            sec_name = self.sec_number_list[index]

            if sec_name.split('_')[0] == 'Inline':
                self.section_type='inline'
            elif sec_name.split('_')[0] == 'Crossline':
                self.section_type='crossline'
            else:
                raise ValueError('Not recognized name: {sec_name}')

            assert self.section_type=='inline' or self.section_type=='crossline',f"Not recognized section type{ self.section_type}"



            if  self.section_type == 'inline':
                section = self.section_volume[:,idx,:]
                if self.task == None or self.task=='segmentation'or self.task=='presentation':
                        label = self.label_volume[:,idx,:]

            elif self.section_type == 'crossline':
                section = self.section_volume[:,:,idx]
                if self.task == None or self.task=='segmentation'or self.task=='presentation':
                        label = self.label_volume[:,:,idx]

            else:
                    raise ValueError(f'Not recognized split type {self.split}')

            #Data Normalization
            section = normalize_1(section)
            if self.split == 'train':
                section = section + np.random.normal(loc=0,scale=0.3)
                section = normalize_1(section)

            if self.section_type == 'inline':
                assert section.shape == (1006,782), f'unexpected shape {section.shape}, should equal to (1006,782) for inline sections'

            elif self.section_type == 'crossline':
                assert section.shape == (1006,590), f'unexpected shape {section.shape}, should equal to (1006,590) for crossline sections'
            else:
                raise ValueError(f"Not recognized shape {self.section_type}")


            if self.task == None or self.task=='segmentation':

                if self.split=='train' or self.split == None:
                    seg_section, seg_mask = random_crop(section, label,final_size=(832,448)) 
                    #add random noise
                    p = np.random.randint(0,2, size=1)
                    if p == 1:
                        seg_section = np.fliplr(seg_section)
                        seg_mask = np.fliplr(seg_mask)

                elif self.split=='test' or self.split=='val':
                    seg_section,seg_mask = center_crop(section, label,final_height=832,final_width=448) 

                else:
                    raise ValueError(f'Not recognized split {self.split}')

                seg_section = np.expand_dims(seg_section, axis=0)
                seg_mask = np.expand_dims(seg_mask, axis=0)

                seg_section_copy = seg_section.copy()
                seg_mask_copy = seg_mask.copy()

                assert seg_section_copy.shape == seg_mask_copy.shape == (1,832,448), f'Expected same shape for section and label (1,448,448), but got: {seg_section_copy.shape} and {seg_mask_copy.shape}'
                return torch.from_numpy(seg_section_copy) , torch.from_numpy(seg_mask_copy), sec_name
            
            
            elif self.task == 'rotation':
                rotated_section, rotation_label = rotate_tensor(section, self.rotations)

                if self.section_type == 'inline':
                    rotated_section = center_crop(rotated_section, final_height=950,final_width=750)
                elif self.section_type == 'crossline':
                    rotated_section = center_crop(rotated_section, final_height=950,final_width=550)
                else:
                    raise ValueError(f'Not recognized section_type: {self.section_type}')

                if self.split=='train' or self.split == None:
                    p = np.random.randint(0,2, size=1)
                    if p == 1:
                        rotated_section = np.fliplr(rotated_section)
                    rotated_section = random_crop(rotated_section,final_size=(832,448))   

                elif self.split=='test' or self.split == 'val':
                    rotated_section = center_crop(rotated_section, final_height=832,final_width=448)

                else:
                    raise ValueError(f'Not recognized split {self.split}')

                rotated_section = np.expand_dims(rotated_section, axis=0)
                rotated_section_copy = rotated_section.copy()
                rotation_label_copy = rotation_label.copy()

                return torch.from_numpy(rotated_section_copy), torch.from_numpy(rotation_label_copy), sec_name


            elif self.task == "jigsaw":

                if self.split=='train'or self.split == None:
                    resized_section = random_crop(section,final_size=(840,420))
                    #add random noise
                    p = np.random.randint(0,2, size=1) #3
                    if p == 1:
                        resized_section = np.fliplr(resized_section)

                elif self.split=='test' or self.split=='val':
                    resized_section = center_crop(section, final_height=840,final_width=420)

                else:
                    raise ValueError(f'Not recognized split {self.split}')

                labels, tiles, labels_orig, tiles_orig,permutation_idx = create_jigsaw(resized_section, self.possible_permutations, size=(256,128),
                                                                                    dataset=self.dataset_name,split=self.split)
               
                return torch.from_numpy(labels), torch.from_numpy(tiles), torch.from_numpy(labels_orig), torch.from_numpy(tiles_orig), permutation_idx, sec_name      


            else:
                raise ValueError(f'Not recognized task: {self.task}') '''


              
        ###############################################################################################################

        
        elif self.dataset_name=='F3_netherlands':
            # Reading items from list.
            sec_idx = self.sections[index]
            sec_name = str(sec_idx).split("/")[-1]
            
            #Reading Images
            section = cv2.imread(sec_idx,-1)
            
            #Casting images to appropiate dtypes
            section = np.array(section).astype(np.float32)
            
            #Data Normalization
            section = normalize_1(section)
            if self.split == 'train':
                section = section + np.random.normal(loc=0,scale=0.3)
                section = normalize_1(section)
            
            #implicitly define section type
            if section.shape == (462,951):
                self.section_type = 'inline'
            elif section.shape == (462,651):
                self.section_type = 'crossline'
            else:
                raise ValueError(f"Not recognized shape {section.shape}")

            assert self.section_type=='inline' or self.section_type=='crossline',f"Not recognized section type{ self.section_type}"

            #Now perform final transformations according to each specific task
            if self.task == None or self.task=='segmentation':
                mask_idx = self.masks_list[index]
                mask = cv2.imread(mask_idx,-1)
                mask = np.array(mask) 

                #First round of cropping to deal with images in standard size
                if self.section_type == 'inline':
                    seg_section, seg_mask = center_crop(section, mask,final_height=450,final_width=900) 

                elif self.section_type == 'crossline':
                    seg_section, seg_mask = crop(section, mask,top=12, left=1,final_height=450,final_width=600) 
                else:
                    raise ValueError(f'Not recognized section_type: {self.section_type}')

                if self.split=='train' or self.split == None:
                    seg_section, seg_mask = random_crop(seg_section, seg_mask,final_size=(448,448)) 

                    p = np.random.randint(0,2, size=1)
                    if p == 1:
                        seg_section = np.fliplr(seg_section)
                        seg_mask = np.fliplr(seg_mask)

                elif self.split=='test' or self.split=='val':
                    seg_section,seg_mask = center_crop(seg_section, seg_mask,final_height=448,final_width=448)

                else:
                    raise ValueError(f'Not recognized split {self.split}')

                seg_section = np.expand_dims(seg_section, axis=0)
                seg_mask = np.expand_dims(seg_mask, axis=0)

                seg_section_copy = seg_section.copy()
                seg_mask_copy = seg_mask.copy()


                return torch.from_numpy(seg_section_copy) , torch.from_numpy(seg_mask_copy), sec_name

            
            elif self.task == "rotation":
                if self.section_type == 'inline':
                    cropped_section = center_crop(section,final_height=450,final_width=900) 

                elif self.section_type == 'crossline':
                    cropped_section = crop(section,top=12, left=1,final_height=450,final_width=600) 
                else:
                    raise ValueError(f'Not recognized section_type: {self.section_type}')
                
                rotated_section, rotation_label = rotate_tensor(cropped_section, self.rotations)

                if self.split=='train' or self.split == None:
                    #rotated_section = rotated_section + np.random.normal(loc=0,scale=0.3)
                    #rotated_section = normalize_1(rotated_section)

                    p = np.random.randint(0,2, size=1)
                    if p == 1:
                        section = np.fliplr(section)
                    rotated_section = random_crop(rotated_section,final_size=(336,336)) 

                elif self.split=='test' or self.split == 'val':
                    rotated_section = center_crop(rotated_section, final_height=336,final_width=336)
                else:
                    raise ValueError(f'Not recognized split {self.split}')

                rotated_section = np.expand_dims(rotated_section, axis=0)

                return torch.from_numpy(rotated_section), torch.from_numpy(rotation_label), sec_name

            elif self.task == "jigsaw":

                if self.section_type == 'inline':
                    cropped_section = center_crop(section,final_height=455,final_width=900) 

                elif self.section_type == 'crossline':
                    cropped_section = crop(section,top=5, left=1,final_height=455,final_width=620) 
                else:
                    raise ValueError(f'Not recognized section_type: {self.section_type}')

                if self.split=='train'or self.split == None:
                    resized_section = random_crop(cropped_section,final_size=(420,618))

                    #add random flip
                    p = np.random.randint(0,2, size=1) 
                    if p == 1:
                        resized_section = np.fliplr(resized_section)

                elif self.split=='test' or self.split=='val':
                    resized_section = center_crop(cropped_section, final_height=420,final_width=618) #tiles 128,192  

                else:
                    raise ValueError(f'Not recognized split {self.split}')

                labels, tiles, labels_orig, tiles_orig,permutation_idx = create_jigsaw(resized_section, self.possible_permutations,
                                                             size=(128,192),dataset=self.dataset_name,split=self.split) 

                return torch.from_numpy(labels), torch.from_numpy(tiles), torch.from_numpy(labels_orig), torch.from_numpy(tiles_orig), permutation_idx, sec_name
            
            else:
                raise ValueError(f'Not recognized task {self.task}')

        else:
            raise ValueError(f'Not recognized dataset {self.dataset_name}')


    def __len__(self):
        return len(self.sections)
