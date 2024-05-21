from .dataset import DataSample, TrainSample, Dataset
from accelerate.logging import get_logger
from datasets import load_dataset, DatasetDict
import random

logger = get_logger(__name__, log_level="INFO")

class ko_wikidata_QA(Dataset):
    def __init__(
        self,
        dataset_name: str = "ko_wikidata_QA",
        split: str = "validation",
        file_path: str = "maywell/ko_wikidata_QA",
        cache_dir: str = "/data/llm/",
        # cache_dir: str = "D:\\huggingface\\cache",
        separator: str = "!@#$%^&*()",
    ):
        self.dataset_name = dataset_name
        self.split = split
        self.data = []
        self.load_data(file_path, cache_dir)
        self.separator = separator

    def __len__(self):
        return len(self.data)

    def load_data(self, file_path: str = None, cache_dir: str = None):
        logger.info(f"Loading ko_wikidata_QA data...")

        # 시드 설정
        random.seed(42)

        raw_datasets = load_dataset(file_path, cache_dir=cache_dir)
        id_ = 0

        train_testsplit = raw_datasets['train'].train_test_split(test_size=0.1, seed=42)
        fix_datasets = DatasetDict({
            'train': train_testsplit['train'],
            'validation': train_testsplit['test']
        })

        for dataset in fix_datasets['train']:
            self.data.append(
                    DataSample(
                        id_=id_,
                        query='Given a question query, retrieve relevant documents that answer the query.' + self.separator + dataset['instruction'],
                        positive=dataset['output'],
                    )
                )
            id_ += 1

        logger.info(f"Loaded {len(self.data)} samples.")

    def __getitem__(self, index):
        sample = self.data[index]
        if self.split == "train":
            return TrainSample(
                texts=[sample.query, sample.positive], label=1.0
            )
        elif self.split == "validation":
            assert False, "Wiki1M does not have a validation split."