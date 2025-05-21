from torch.utils.data import Dataset, DataLoader
import torch
import lightning as L
import pandas as pd


#Creating a custom dataset class in pytorch to store the transliteration dataset.
class TransliterationDataset(Dataset):
    def __init__(self, dataframe, source_vocab, target_vocab):
        self.dataframe = dataframe
        self.source_vocab = source_vocab
        self.target_vocab = target_vocab

    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        src_words = [ self.source_vocab[chr] if chr in self.source_vocab else self.source_vocab["<UNK>"] for chr in self.dataframe.iloc[idx]["source"]]

        # Similarly, for the target words        
        tgt_words = [self.target_vocab[chr] if chr in self.target_vocab else self.target_vocab["<UNK>"] for chr in self.dataframe.iloc[idx]["target"]]
        # Add <SOW> and <EOW> tokens to the target words
        tgt_words = [self.target_vocab["<SOW>"]] + tgt_words + [self.target_vocab["<EOW>"]]

        return torch.LongTensor(src_words), torch.LongTensor(tgt_words) 


# Since we are using lightning module, Let's define the Lightning data module to handle the dataset. 
# Within this class we have defined dataloader for all train, dev and test dataset. 
# Also this same module takes care for vocabalary builder and tokenization
# This will helps us modularise the code better
class TrasnliterationDataModule(L.LightningDataModule):
    def __init__(self, data_dir, batch_size=32):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size

        
        # Load the train dataset
        self.train_df = pd.read_csv(
            f"{self.data_dir}/hi.translit.sampled.train.tsv",
            sep="\t",
            names=["target", "source", "attestations"],
            header=None,
            keep_default_na=False, na_values=[]
        )
        # Drop the attestations column
        self.train_df.drop(columns=["attestations"], inplace=True)

        # Let's load the dev set as well
        # We will use the dev set for validation
        self.dev_df = pd.read_csv(
            f"{self.data_dir}/hi.translit.sampled.dev.tsv",
            sep="\t",
            names=["target", "source", "attestations"],
            header=None,
            keep_default_na=False, na_values=[]
        )
        # Drop the attestations column
        self.dev_df.drop(columns=["attestations"], inplace=True)


        #Let's load the test set as well
        # We will use the test set for evaluation
        self.test_df = pd.read_csv(
            f"{self.data_dir}/hi.translit.sampled.test.tsv",
            sep="\t",
            names=["target", "source", "attestations"],
            header=None,
            keep_default_na=False, na_values=[]
        )
        # Drop the attestations column
        self.test_df.drop(columns=["attestations"], inplace=True)

        # Create vocabularies for source and target languages
        self.source_vocab, self.source_chr_to_idx, self.source_idx_to_char = self.build_vocab(self.train_df['source'].values)
        self.target_vocab, self.target_chr_to_idx, self.target_idx_to_char = self.build_vocab(self.train_df['target'].values)


    def prepare_data(self):
        '''
        According the the Lightning documentation, this method is used to download and prepare the data.
        In our case, we are not downloading any data, dataset can be found at the given data_dir path, but we are preparing the data
        '''
        
        self.train_dataset = self.create_dataset(self.train_df)
        self.dev_dataset = self.create_dataset(self.dev_df)
        self.test_dataset = self.create_dataset(self.test_df)

    def build_vocab(self, words):
        '''
        This method is used to build the vocab for the given data
        :param data: The data to build the vocab for
        :return: The vocab for the given data
        '''
        vocab = set()
        for word in words:
            for char in word:
                vocab.add(char)

        # Adding special tokens in the vocab.
        vocab.add("<UNK>")
        vocab.add("<EOW>")
        vocab.add("<SOW>")

        # Sort the vocab to get the same order every time
        vocab = sorted(vocab)
        
        chr_to_idx_map = { chr : idx+1 for idx, chr in enumerate(vocab) }
        idx_to_chr_map = { idx+1 : chr for idx, chr in enumerate(vocab) }

        chr_to_idx_map["<PAD>"] = 0
        idx_to_chr_map[0] = "<PAD>"

        vocab.append("<PAD>")
        
        return vocab, chr_to_idx_map, idx_to_chr_map

    def create_dataset(self, dataframe):
        '''
        This method is used to create the dataset for the given data
        :param dataframe: The dataframe to create the dataset for
        :return: The dataset for the given data
        '''
        return TransliterationDataset(dataframe, self.source_chr_to_idx, self.target_chr_to_idx)
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size = self.batch_size, shuffle=True, collate_fn = self.collate_fn, num_workers=3)
    
    def val_dataloader(self):
        return DataLoader(self.dev_dataset, batch_size = self.batch_size, shuffle=False, collate_fn = self.collate_fn, num_workers=3)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size = self.batch_size, shuffle=False, collate_fn = self.collate_fn, num_workers=3)
    
    def collate_fn(self, batch):
        '''
        This method is used to collate the data into batches
        :param batch: The batch to collate
        :return: The collated batch
        '''
        src_words, tgt_words = zip(*batch)
        
        # Pad the source and target words
        src_words = torch.nn.utils.rnn.pad_sequence(src_words, batch_first=True, padding_value=0)
        tgt_words = torch.nn.utils.rnn.pad_sequence(tgt_words, batch_first=True, padding_value=0)

        return src_words, tgt_words
    
#Now, Let's define the helper function that will convert token back to the words.
def decode_sequence(tensor, chr2idx, idx2char):
    # Here, chr2idx, and idx2chr both are dictionaries that holds the mapping from characters to index and vice versa
    chars = []
    for idx in tensor:
        if idx.item() in [chr2idx['<EOW>'], chr2idx['<PAD>']]:
            break
        chars.append(idx2char.get(idx.item(), '<UNK>'))
    return ''.join(chars)
