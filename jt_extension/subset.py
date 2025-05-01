from jittor.dataset import Dataset
from typing import Sequence, Any

class Subset(Dataset):
    def __init__(self, dataset: Dataset, indices: Sequence[int]) -> None:
        super().__init__()

        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, index: int) -> Any:

        if not (0 <= index < len(self.indices)):
             raise IndexError(f"Subset index out of range: {index}")

        original_index = self.indices[index]

        return self.dataset[original_index]

    def __len__(self) -> int:
        return len(self.indices)