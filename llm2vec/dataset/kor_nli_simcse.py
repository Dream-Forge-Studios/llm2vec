from .dataset import DataSample, TrainSample, Dataset
from accelerate.logging import get_logger
from datasets import load_dataset

logger = get_logger(__name__, log_level="INFO")

class kor_nli_simcse(Dataset):
    def __init__(
        self,
        dataset_name: str = "kor_nli_simcse",
        split: str = "validation",
        file_path: str = "dkoterwa/kor_nli_simcse",
        cache_dir: str = "/data/llm/",
        # cache_dir: str = "D:\\huggingface\\cache",
    ):
        self.dataset_name = dataset_name
        self.split = split
        self.data = []
        self.load_data(file_path, cache_dir)

    def __len__(self):
        return len(self.data)

    def load_data(self, file_path: str = None, cache_dir: str = None):
        logger.info(f"Loading kor_nli_simcse data...")

        raw_datasets = load_dataset(file_path, "default", cache_dir=cache_dir)
        id_ = 0
        for dataset in raw_datasets['train']:
            self.data.append(
                    DataSample(
                        id_=id_,
                        query=dataset['premise'],
                        positive=dataset['entailment'],
                        negative=dataset['contradiction'],
                    )
                )
            id_ += 1

        logger.info(f"Loaded {len(self.data)} samples.")

    def __getitem__(self, index):
        sample = self.data[index]
        if self.split == "train":
            return TrainSample(
                texts=[sample.query, sample.positive, sample.negative], label=1.0
            )
        elif self.split == "validation":
            assert False, "Wiki1M does not have a validation split."