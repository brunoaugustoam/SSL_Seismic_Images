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
from aux_functions import *


class SeismicDataset(data.Dataset):
    def __init__(self, root, dataset_name, split,train_type='sup_ssl', task=None, n_few_shot=None,cross_val=False, fold=None,present_sec='inline'):
        self.root = root       
        self.split = split.lower()
        self.dataset_name =  dataset_name
        self.train_type = train_type
        self.partition = [0.60,0.20,0.20]      # split1: [0.50,0.25,0.25]  split2: [0.70,0.15,0.15] split3:[0.60,0.20,0.20]

        self.task = task
        self.rotations = [-8,-4,0,4,8]
        self.cell_size = (14,14)
        self.hog_orientations = 8 #8bins
        self.cross_val = cross_val
        self.fold = fold

        if self.train_type == 'few_shot' or self.train_type == 'fine_tune':
            assert n_few_shot != None, 'None Type is not valid'
            self.n_few_shot = n_few_shot #for now, later change to percentage or others

        if self.cross_val:
            assert fold != None, 'None Type is not valid'
            self.fold = fold #for now, later change to percentage or others
            assert self.fold >= 0 and self.fold <= 4, f'Not Valid Fold: {self.fold}'
        
        
        assert self.dataset_name=='F3_netherlands' or self.dataset_name=='Parihaka_NZPM',f"Not recognized dataset {self.dataset_name}"
        assert self.split=='train' or self.split=='val' or self.split=='test' or self.split=='presentation' , f"Not recognized split type {self.split}"
        assert self.task== None or self.task=='segmentation' or self.task=='rotation' or self.task=='hog' or self.task=='jigsaw' or self.task =='presentation' or self.task=='inpainting', f"Not recognized task{self.task}"
        assert self.train_type=='sup_ssl' or self.train_type=='few_shot'or self.train_type=='fine_tune' or self.train_type=='presentation', f"Not recognized train type type {self.train_type}"
        
        if self.split == 'presentation':
                self.present_sec = present_sec
                #assert self.present_sec == 'inline' or self.present_sec == 'crossline', f'Invalid presentation section type {self.present_sec}'
                self.task = 'presentation'

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
        ##### Netherlands F3   ####### SPECIFIC TRANSFORMATIONS FOR EACH DATASET ########
        
        # Specify Paths
        if self.dataset_name=='F3_netherlands':
            
            self.masks_path = 'masks/'
            self.data_path = 'sections_joined/'

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

            seed = 42 #np.random.randint(3125161651,size=1)
            np.random.seed(seed) ; sections_list = list(np.random.permutation(sections_list))
            np.random.seed(seed) ; masks_list = list(np.random.permutation(masks_list))
            np.random.seed(seed) ; sec_number_list = list(np.random.permutation(sec_number_list))

            if self.cross_val == True:
                if self.split=='presentation' or self.task=='presentation':
                    raise ValueError(f"Cross Validation and presentation split/task togheter makes no sense")

                sections_list,masks_list,sec_number_list = make_cross_val_folds_f3(sections_list,masks_list,sec_number_list,self.fold,self.split,self.partition[1])
                # AQUI PRECISA ARRUMAR PRA FOLD DEVOLVER N SECTIONS DE CADA FOLD. DA FORMA COMO ESTA, DEVOLVE AS 9 PRIMEIRA
                #PARA APROVEITAR, PODE PEGAR ESSA LISTA JA COM AS FOLDS, FAZ UMA NOVA PERMUTACAO E AI SEM PEGA AS 9 PRIMEIRAS
                seed = 42 #np.random.randint(3125161651,size=1)
                np.random.seed(seed) ; sections_list = list(np.random.permutation(sections_list))
                np.random.seed(seed) ; masks_list = list(np.random.permutation(masks_list))
                np.random.seed(seed) ; sec_number_list = list(np.random.permutation(sec_number_list))
                #####Vai precisar arrumar a verificacao das folds :/
                verify_folds_f3(self.split, self.fold, sec_number_list)

            else:

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

                elif self.split=='presentation': #secoes representativas do F3 selecionadas a mao - todas pertencem ao conjunto de teste
                    assert self.task == 'presentation', f'presentation requires both task and split to be presentation'
                    #Crossline: 886, 1090
                    #Inline: 295, 587
                    if self.present_sec == 'inline':
                        self.partitions = [0,1,2,3,4]
                        sections_list = ['/mnt/DADOS_CERGY_1/bruno/F3_netherlands/sections_joined/inline_295.tiff']*len(self.partitions) #,'/mnt/DADOS_CERGY_1/bruno/F3_netherlands/sections_joined/inline_587.tiff']
                        masks_list = [  '/mnt/DADOS_CERGY_1/bruno/F3_netherlands/masks/inline_295_mask.png']*len(self.partitions) # '/mnt/DADOS_CERGY_1/bruno/F3_netherlands/masks/inline_587_mask.png']
                        sec_number_list = ['inline_295']*len(self.partitions) #'inline_587'
                        
                    elif self.present_sec == 'crossline':
                        self.partitions = [0,1,2]
                        masks_list = ['/mnt/DADOS_CERGY_1/bruno/F3_netherlands/masks/crossline_886_mask.png']*len(self.partitions)#'/mnt/DADOS_CERGY_1/bruno/F3_netherlands/masks/crossline_1090_mask.png']*len(self.partitions)
                        sections_list = [ '/mnt/DADOS_CERGY_1/bruno/F3_netherlands/sections_joined/crossline_886.tiff']*len(self.partitions) #'/mnt/DADOS_CERGY_1/bruno/F3_netherlands/sections_joined/crossline_1090.tiff',]*len(self.partitions)
                        sec_number_list = ['crossline_886']*len(self.partitions) #'crossline_1090']*len(self.partitions)
                    
                    elif self.present_sec == 'inline_587':
                        self.partitions = [0,1,2,3,4]
                        sections_list = ['/mnt/DADOS_CERGY_1/bruno/F3_netherlands/sections_joined/inline_587.tiff']*len(self.partitions) #,'/mnt/DADOS_CERGY_1/bruno/F3_netherlands/sections_joined/inline_587.tiff']
                        masks_list = [  '/mnt/DADOS_CERGY_1/bruno/F3_netherlands/masks/inline_587_mask.png']*len(self.partitions) # '/mnt/DADOS_CERGY_1/bruno/F3_netherlands/masks/inline_587_mask.png']
                        sec_number_list = ['inline_587']*len(self.partitions) #'inline_587'    
                    
                    else:
                        raise ValueError(f'Not recognized present_sec type {self.present_sec}')

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
                #for item in sec_index:
                #    sec_few.append(os.path.join(self.data_path,sections_list[item]))
                #    mask_few.append(masks_list[item])
                #    sec_number_few.append(sec_number_list[item])

                return sec_few, mask_few,sec_number_few

            else:
                return sections_list, masks_list, sec_number_list

        #### New Zealand Parihaka  ###### SPECIFIC TRANSFORMATIONS FOR EACH DATASET ########
        #######################################################################################################
        elif self.dataset_name=='Parihaka_NZPM':

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
            if self.task == None or self.task=='segmentation'or self.task=='presentation':
                self.label_volume = segyio.tools.cube(self.masks_path)
                for i in range(0,6):
                    self.label_volume = np.where(self.label_volume ==i+1,i,self.label_volume)

                assert self.label_volume.shape == self.section_volume.shape, f'Expected same shape for section and label, but got: {self.label_volume.shape} and {self.section_volume.shape}'

            self.section_volume =  np.transpose(self.section_volume, (2, 0, 1))
            
            if self.task == None or self.task=='segmentation' or self.task=='presentation':
                self.label_volume =  np.transpose(self.label_volume, (2, 0, 1))
                assert self.label_volume.shape == self.section_volume.shape, f'Expected same shape for section and label (1006,590,782), but got: {self.label_volume.shape} and {self.section_volume.shape}'

            inline = np.arange(0,self.section_volume.shape[1])
            crossline = np.arange(0,self.section_volume.shape[2])

            seed = 42 #np.random.randint(3125161651,size=1)
            np.random.seed(seed) ; inline = list(np.random.permutation(inline))
            np.random.seed(seed) ; crossline = list(np.random.permutation(crossline))

            if self.cross_val == True:
                if self.split=='presentation' or self.task=='presentation':
                    raise ValueError(f"Cross Validation and presentation split/task togheter makes no sense")

                inline_name = ['Inline_' + z for z in [str(i) for i in inline]]
                crossline_name = ['Crossline_' + z for z in [str(i) for i in crossline]]

                sections = np.hstack(np.array([inline, crossline]))
                sec_name = inline_name + crossline_name

                #here we must permute again beacuse the folds follow the order of disposition of the data
                #outside cross validation it is not a problem because we are getting data from this list with
                #shuffling option turned on
                seed = 42 #np.random.randint(3125161651,size=1)
                np.random.seed(seed) ; sections = list(np.random.permutation(sections))
                np.random.seed(seed) ; sec_name = list(np.random.permutation(sec_name))

                sections,sec_name = make_cross_val_folds_parih(sections,sec_name,self.fold,self.split,self.partition[1])

                ##Here permuting again so the folds are mixed for the few shot
                seed = 42 #np.random.randint(3125161651,size=1)
                np.random.seed(seed) ; sections = list(np.random.permutation(sections))
                np.random.seed(seed) ; sec_name = list(np.random.permutation(sec_name))

                verify_folds_parih(self.split, self.fold, sec_name)

                if self.split=='train':
                    if self.train_type == 'few_shot' or self.train_type == 'fine_tune':
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
                            inline_name = ['Inline_' + z for z in [str(i) for i in inline]]
                            crossline_name = ['Crossline_' + z for z in [str(i) for i in crossline]]
                            sections = np.hstack(np.array([inline, crossline]))
                            sec_name = inline_name + crossline_name
                            return sections,sec_name
                            

            else:

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

                elif self.split=='presentation': #secoes representativas do F3 selecionadas a mao - todas pertencem ao conjunto de teste
                    #Crossline: 
                    #Inline: 
                    if self.present_sec == 'inline':
                        self.partitions = [0,1,2,3,4,5]
                        sections = [552]*len(self.partitions) #[552]*len(self.partitions)
                        
                    elif self.present_sec == 'crossline':
                        self.partitions = [0,1,2,3]
                        sections = [602]*len(self.partitions) #[602]*len(self.partitions)
                        
                    else:
                        raise ValueError(f'Not recognized present_sec type {self.present_sec}')

                else:
                    raise ValueError(f'Not recognized split type {self.split}')
                
                if self.split=='presentation':
                    if self.present_sec == 'inline':
                        sec_name = ['Inline_' + z for z in [str(i) for i in sections]]
                    elif self.present_sec == 'crossline':
                        sec_name = ['Crossline_' + z for z in [str(i) for i in sections]]

                else:
                    inline_name = ['Inline_' + z for z in [str(i) for i in inline]]
                    crossline_name = ['Crossline_' + z for z in [str(i) for i in crossline]]
                    sections = np.hstack(np.array([inline, crossline]))
                    sec_name = inline_name + crossline_name

            return sections,sec_name
        
        else:
            raise ValueError(f'Not recognized dataset {self.dataset_name}')
        
        
    def __getitem__(self, index):

        if self.dataset_name=='Parihaka_NZPM':

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
            
            elif self.task == 'presentation': ####### precisa arrumar TUDO
                if self.section_type == 'inline':
                    present_section, present_mask = center_crop(section, label,final_height=1000,final_width=600) 
                elif self.section_type == 'crossline':
                    present_section, present_mask = crop(section, label,top=12, left=1,final_height=1000,final_width=600) 
                else:
                    raise ValueError(f'Not recognized section_type: {self.section_type}')

                #for slicex in range(len(self.partitions)):
                if self.section_type=='inline':
                    if self.partitions[index]==0:
                        present_section, present_mask = crop(present_section, present_mask,top=0, left=0,final_height=832,final_width=448) 
                    elif self.partitions[index]==1:
                        present_section, present_mask = crop(present_section, present_mask,top=0, left=167,final_height=832,final_width=448) 
                    elif self.partitions[index]==2:
                        present_section, present_mask = crop(present_section, present_mask,top=0, left=334,final_height=832,final_width=448) 
                    elif self.partitions[index]==3:
                        present_section, present_mask = crop(present_section, present_mask,top=174, left=0,final_height=832,final_width=448) 
                    elif self.partitions[index]==4:
                        present_section, present_mask = crop(present_section, present_mask,top=174, left=167,final_height=832,final_width=448) 
                    elif self.partitions[index]==5:
                        present_section, present_mask = crop(present_section, present_mask,top=174, left=334,final_height=832,final_width=448)
                    else:
                        raise ValueError('Unexpected index')

                elif self.section_type=='crossline':
                    if self.partitions[index]==0:
                        present_section, present_mask = crop(present_section, present_mask,top=0, left=0,final_height=832,final_width=448) 
                    elif self.partitions[index]==1:
                        present_section, present_mask = crop(present_section, present_mask,top=0, left=142,final_height=832,final_width=448) 
                    elif self.partitions[index]==2:
                        present_section, present_mask = crop(present_section, present_mask,top=174, left=0,final_height=832,final_width=448) 
                    elif self.partitions[index]==3:
                        present_section, present_mask = crop(present_section, present_mask,top=174, left=142,final_height=832,final_width=448) 
                    else:
                        raise ValueError('Unexpected index')

                present_section = np.expand_dims(present_section, axis=0)
                present_mask = np.expand_dims(present_mask, axis=0)

                present_section_copy = present_section.copy()
                present_mask_copy = present_mask.copy()

                return torch.from_numpy(present_section_copy) , torch.from_numpy(present_mask_copy), sec_name


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


            elif self.task == 'hog':
                if self.split=='train' or self.split == None:
                    resized_section = random_crop(section,final_size=(832,448))
                    p = np.random.randint(0,2, size=1)
                    if p == 1:
                        resized_section = np.fliplr(resized_section)

                elif self.split=='test' or self.split == 'val':
                    resized_section = center_crop(section, final_height=832,final_width=448)
                else:
                    raise ValueError(f'Not recognized split {self.split}')
                
                pil_section = T.ToPILImage()(resized_section)

                out, hog_image = hog(pil_section, orientations=self.hog_orientations, 
                    pixels_per_cell=self.cell_size, cells_per_block=(1, 1), 
                    feature_vector=False, visualize=True)

                hog_image = np.array(hog_image).astype(np.float32)

                resized_section = np.expand_dims(resized_section, axis=0)
                hog_image = np.expand_dims(hog_image, axis=0)

                out = np.array(out).astype(np.float32)
                label_out = out.copy()
                resized_section_out = resized_section.copy()

                return torch.from_numpy(resized_section_out), torch.from_numpy(hog_image), torch.from_numpy(label_out), sec_name

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

            elif self.task=='inpainting':
                if self.split=='train'or self.split == None:
                    resized_section = random_crop(section,final_size=(832,448))
                    #add random flip
                    p = np.random.randint(0,2, size=1) #3
                    if p == 1:
                        resized_section = np.fliplr(resized_section)

                    cropped_section,cropped_tile, height_idx,width_idx,final_size = random_crop_inpainting(resized_section,final_size=(192*2,192),
                                                                                                            plot=False,dataloader_crop=True)


                elif self.split=='test' or self.split=='val':
                    resized_section = center_crop(section, final_height=832,final_width=448)
                    cropped_section,cropped_tile,height_idx,width_idx,final_size  = center_crop_inpainting(resized_section,final_size=(192*2,192),
                                                                                                                    plot=False,dataloader_crop=True)

                else:
                    raise ValueError(f'Not recognized split {self.split}')

                cropped_section = np.expand_dims(cropped_section, axis=0)
                cropped_tile = np.expand_dims(cropped_tile, axis=0)
                resized_section = np.expand_dims(resized_section, axis=0)

                cropped_section_copy = cropped_section.copy()
                cropped_tile_copy = cropped_tile.copy()
                resized_section = resized_section.copy()

                return torch.from_numpy(resized_section), torch.from_numpy(cropped_section_copy), torch.from_numpy(cropped_tile_copy), height_idx,width_idx,final_size,sec_name


            else:
                raise ValueError(f'Not recognized task: {self.task}')


              
        ###################################################################################################################3

        
        elif self.dataset_name=='F3_netherlands':
            # Reading items from list.
            sec_idx = self.sections[index]
            sec_name = str(sec_idx).split("/")[-1]
            
            #Reading Images
            section = cv2.imread(sec_idx,-1)
            #mask = cv2.imread(mask_idx,-1)
            
            #Casting images to appropiate dtypes
            section = np.array(section).astype(np.float32)
            
            #Data Normalization
            section = normalize_1(section)
            if self.split == 'train':
                section = section + np.random.normal(loc=0,scale=0.3)
            #    section = section + np.random.normal(loc=np.mean(section),scale=np.std(section))
                section = normalize_1(section)
            #mask = np.array(mask) #.astype(np.int64)
            
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
                mask = np.array(mask) #.astype(np.int64)

                #First round of cropping to deal with images in standard size
                #always keeping as np array, only convert to torchTensor at return
                if self.section_type == 'inline':
                    seg_section, seg_mask = center_crop(section, mask,final_height=450,final_width=900) 

                elif self.section_type == 'crossline':
                    seg_section, seg_mask = crop(section, mask,top=12, left=1,final_height=450,final_width=600) 
                else:
                    raise ValueError(f'Not recognized section_type: {self.section_type}')

                if self.split=='train' or self.split == None:
                    seg_section, seg_mask = random_crop(seg_section, seg_mask,final_size=(448,448)) 
                    #seg_section = seg_section + np.random.normal(loc=0,scale=0.3)
                    #seg_section = normalize_1(seg_section)
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

            elif self.task == 'presentation':
                mask_idx = self.masks_list[index]
                mask = cv2.imread(mask_idx,-1)
                mask = np.array(mask) #.astype(np.int64)
                if self.section_type == 'inline':
                    present_section, present_mask = center_crop(section, mask,final_height=448,final_width=896) 
                elif self.section_type == 'crossline':
                    present_section, present_mask = crop(section, mask,top=12, left=1,final_height=448,final_width=600) 
                else:
                    raise ValueError(f'Not recognized section_type: {self.section_type}')

                #for slicex in range(len(self.partitions)):
                if self.section_type=='inline':
                    if self.partitions[index]==0:
                        present_section, present_mask = crop(present_section, present_mask,top=0, left=0,final_height=448,final_width=448) 
                    elif self.partitions[index]==1:
                        present_section, present_mask = crop(present_section, present_mask,top=0, left=112,final_height=448,final_width=448) 
                    elif self.partitions[index]==2:
                        present_section, present_mask = crop(present_section, present_mask,top=0, left=224,final_height=448,final_width=448) 
                    elif self.partitions[index]==3:
                        present_section, present_mask = crop(present_section, present_mask,top=0, left=336,final_height=448,final_width=448) 
                    elif self.partitions[index]==4:
                        present_section, present_mask = crop(present_section, present_mask,top=0, left=448,final_height=448,final_width=448) 
                    else:
                        raise ValueError('Unexpected index')
                elif self.section_type=='crossline':
                    if self.partitions[index]==0:
                        present_section, present_mask = crop(present_section, present_mask,top=0, left=0,final_height=448,final_width=448) 
                    elif self.partitions[index]==1:
                        present_section, present_mask = crop(present_section, present_mask,top=0, left=76,final_height=448,final_width=448) 
                    elif self.partitions[index]==2:
                        present_section, present_mask = crop(present_section, present_mask,top=0, left=152,final_height=448,final_width=448) 
                    else:
                        raise ValueError('Unexpected index')

                present_section = np.expand_dims(present_section, axis=0)
                present_mask = np.expand_dims(present_mask, axis=0)

                present_section_copy = present_section.copy()
                present_mask_copy = present_mask.copy()

                return torch.from_numpy(present_section_copy) , torch.from_numpy(present_mask_copy), sec_name


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

            elif self.task == "hog":
                if self.section_type == 'inline':
                    cropped_section = center_crop(section,final_height=450,final_width=900) 

                elif self.section_type == 'crossline':
                    cropped_section = crop(section,top=10, left=1,final_height=451,final_width=600) 
                else:
                    raise ValueError(f'Not recognized section_type: {self.section_type}')

                if self.split=='train' or self.split == None:
                    resized_section = random_crop(cropped_section,final_size=(448,448))
                    #resized_section = resized_section + np.random.normal(loc=0,scale=0.3)
                    #resized_section = normalize_1(resized_section)

                    p = np.random.randint(0,2, size=1)
                    if p == 1:
                        resized_section = np.fliplr(resized_section)

                elif self.split=='test' or self.split == 'val':
                    resized_section = center_crop(cropped_section, final_height=448,final_width=448)
                else:
                    raise ValueError(f'Not recognized split {self.split}')
                
                pil_section = T.ToPILImage()(resized_section)

                out, hog_image = hog(pil_section, orientations=self.hog_orientations, 
                    pixels_per_cell=self.cell_size, cells_per_block=(1, 1), 
                    feature_vector=False, visualize=True)

                hog_image = np.array(hog_image).astype(np.float32)

                resized_section = np.expand_dims(resized_section, axis=0)
                hog_image = np.expand_dims(hog_image, axis=0)

                out = np.array(out).astype(np.float32)
                label_out = out.copy()
                resized_section_out = resized_section.copy()

                return torch.from_numpy(resized_section_out), torch.from_numpy(hog_image), torch.from_numpy(label_out), sec_name


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
                    p = np.random.randint(0,2, size=1) #3
                    if p == 1:
                        resized_section = np.fliplr(resized_section)
   

                    

                elif self.split=='test' or self.split=='val':
                    resized_section = center_crop(cropped_section, final_height=420,final_width=618) #tiles 128,192  

                else:
                    raise ValueError(f'Not recognized split {self.split}')

                labels, tiles, labels_orig, tiles_orig,permutation_idx = create_jigsaw(resized_section, self.possible_permutations,
                                                             size=(128,192),dataset=self.dataset_name,split=self.split) 

                return torch.from_numpy(labels), torch.from_numpy(tiles), torch.from_numpy(labels_orig), torch.from_numpy(tiles_orig), permutation_idx, sec_name
            

            elif self.task=='inpainting':
                if self.split=='train'or self.split == None:
                    resized_section = random_crop(section,final_size=(448,448))
                     #add random flip
                    p = np.random.randint(0,2, size=1) #3
                    if p == 1:
                        resized_section = np.fliplr(resized_section)

                    cropped_section,cropped_tile, height_idx,width_idx,final_size = random_crop_inpainting(resized_section,final_size=(128,128),
                                                                                                            plot=False,dataloader_crop=True)

                elif self.split=='test' or self.split=='val':
                    resized_section = center_crop(section, final_height=448,final_width=448)
                    cropped_section,cropped_tile,height_idx,width_idx,final_size  = center_crop_inpainting(resized_section,final_size=(128,128),
                                                                            plot=False,dataloader_crop=True)

                else:
                    raise ValueError(f'Not recognized split {self.split}')

                cropped_section = np.expand_dims(cropped_section, axis=0)
                cropped_tile = np.expand_dims(cropped_tile, axis=0)
                resized_section = np.expand_dims(resized_section, axis=0)

                cropped_section_copy = cropped_section.copy()
                cropped_tile_copy = cropped_tile.copy()
                resized_section = resized_section.copy()

                return torch.from_numpy(resized_section), torch.from_numpy(cropped_section_copy), torch.from_numpy(cropped_tile_copy), height_idx,width_idx,final_size,sec_name
            else:
                raise ValueError(f'Not recognized task {self.task}')

        else:
            raise ValueError(f'Not recognized dataset {self.dataset_name}')


    def __len__(self):
        return len(self.sections)