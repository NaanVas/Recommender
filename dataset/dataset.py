import torch
from torch.utils.data import Dataset

class RatingsDataset(Dataset):
    def __init__(self, ratings_df):
        self.users = torch.tensor(ratings_df['user_id'].values, dtype=torch.long)
        self.itens = torch.tensor(ratings_df['item_id'].values, dtype=torch.long)
        self.ratings = torch.tensor(ratings_df['rating'].values, dtype=torch.float)

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        return self.users[idx], self.itens[idx], self.ratings[idx]