from ..dataset import E5Data, kor_nli, Wiki1M, wikipedia_ko_for_simcse, kor_nli_for_simcse


def load_dataset(dataset_name, split="validation", file_path=None, **kwargs):
    """
    Loads a dataset by name.

    Args:
        dataset_name (str): Name of the dataset to load.
        split (str): Split of the dataset to load.
        file_path (str): Path to the dataset file.
    """
    dataset_mapping = {
        "E5": E5Data,
        "kor_nli": kor_nli,
        "Wiki1M": Wiki1M,
        "wikipedia_ko_for_simcse": wikipedia_ko_for_simcse,
        "kor_nli_for_simcse": kor_nli_for_simcse,
    }

    if dataset_name not in dataset_mapping:
        raise NotImplementedError(f"Dataset name {dataset_name} not supported.")

    if split not in ["train", "validation", "test"]:
        raise NotImplementedError(f"Split {split} not supported.")

    return dataset_mapping[dataset_name](
        split=split, file_path=file_path, **kwargs
    )
