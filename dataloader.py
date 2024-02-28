
import os
from torch.utils.data import Dataset
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision.transforms import ToTensor, Compose

def metadata_build(df_train): 
    df_train['GENDER'] = df_train.GENDER.apply(lambda x: int(x == 'F'))
    df_train['DOB'] = df_train['DOB'].apply(lambda x: x.replace("-", "/"))
    df_train['AGE'] = df_train['DOB'].apply(
        lambda x: 2023-int(x.split("/")[-1]))
    return df_train


class MILDataset(Dataset):
    def __init__(self, root_dir, df, transform=None):
        """
        Args:
            root_dir (string): Directory with all the bags.
            labels_df (DataFrame): DataFrame containing labels and other information.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        df = metadata_build(df)
        self.root_dir = root_dir
        self.labels_df = df
        self.transform = transform
        self.bag_paths = [os.path.join(root_dir, o) for o in df.ID if os.path.isdir(os.path.join(root_dir, o))]
        self.bag_paths.sort()
        self.bag_ids = [os.path.basename(o) for o in self.bag_paths]

        

    def __len__(self):
        return len(self.bag_paths)

    def __getitem__(self, idx):
        bag_id = self.bag_ids[idx]
        bag_path = self.bag_paths[idx]

        instances = []

        for instance_name in os.listdir(bag_path):

            instance_path = os.path.join(bag_path, instance_name)
            instance_image = Image.open(instance_path)



            if self.transform:
                instance_image = self.transform(instance_image)

            instances.append(instance_image.unsqueeze(0))
        features={}
        features['instances']=torch.cat(instances)

        # Retrieve the label for the current bag
   
        label = self.labels_df[self.labels_df['ID'] == bag_id]['LABEL'].values[0]
        features['age'] = self.labels_df[self.labels_df['ID'] == bag_id]['AGE'].values[0]
        features['lymph_count'] = self.labels_df[self.labels_df['ID'] == bag_id]['LYMPH_COUNT'].values[0]
        features['gender'] = self.labels_df[self.labels_df['ID'] == bag_id]['GENDER'].values[0]
        features['id'] = bag_id

        # Convert label to tensor if necessary, depending on your model's requirements
        return features, label
