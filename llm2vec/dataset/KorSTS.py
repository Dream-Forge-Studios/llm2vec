from .dataset import DataSample, TrainSample, Dataset
from accelerate.logging import get_logger

logger = get_logger(__name__, log_level="INFO")

class KorSTS(Dataset):
    def __init__(
        self,
        dataset_name: str = "KorSTS",
        split: str = "validation",
        # file_path: str = "D:/KorSTS/sts-train.tsv",
        file_path: str = "/data/KorSTS/sts-train.tsv",
    ):
        self.dataset_name = dataset_name
        self.split = split
        self.data = []
        self.load_data(file_path)

    def __len__(self):
        return len(self.data)

    def load_data(self, file_path: str = None):
        def scale_value(original_value, original_min, original_max, new_min, new_max):
            # 선형 변환 공식 적용
            return ((original_value - original_min) / (original_max - original_min)) * (new_max - new_min) + new_min

        logger.info(f"Loading KorSTS data from {file_path}...")
        id_ = 0
        with open(file_path, "r", encoding="utf-8") as f:
            next(f)
            for line in f:
                temps = line.strip().split('\t')
                scaled_value = scale_value(float(temps[4]), 0, 5, 0, 1)
                self.data.append(
                    DataSample(
                        id_=id_,
                        query=temps[5],
                        positive=temps[6],
                        negative=scaled_value,
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
