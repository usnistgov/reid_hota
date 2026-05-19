import numpy as np
from scipy.optimize import linear_sum_assignment

from .hota_errors import DuplicateIDError


class CostMatrixData:
    """Poor-man's sparse cost matrix.

    Holds a dense cost matrix plus the id arrays for its two axes, and lazily
    builds id→index maps and Hungarian-assignment results when callers ask for
    them. All non-init state is allocated in `__init__` so attribute lookups
    are predictable for downstream consumers (and AI tooling).
    """

    def __init__(self, i_ids: np.ndarray, j_ids: np.ndarray, cost_matrix: np.ndarray, video_id: str, frame: int):
        self.i_ids = i_ids                # reference IDs along the y axis
        self.j_ids = j_ids                # comparison IDs along the x axis
        self.cost_matrix = cost_matrix
        self.video_id = video_id
        self.frame = frame
        # Lazy state — populated on first use by the helpers below. Kept None
        # by default so workers don't pay the dict-construction cost during
        # pickle/unpickle when they won't read it.
        self._ref_id2idx_map: dict | None = None
        self._comp_id2idx_map: dict | None = None
        # Hungarian-assignment outputs — set by construct_assignment().
        self.match_rows: np.ndarray | None = None
        self.match_cols: np.ndarray | None = None
        self.ref2comp_idx_map: dict[int, int] | None = None
        self.ref2comp_id_map: dict[int, int] | None = None

    def get_cost(self, i: np.dtype[np.object_], j: np.dtype[np.object_]) -> float:
        """Get the cost matrix value at coordinate (i,j).

        Args:
            i: Reference ID (not index)
            j: Comparison ID (not index)

        Returns:
            Cost value at the specified coordinate
        """
        # Find indices of i,j in the id arrays
        i_idx = np.where(self.i_ids == i)[0]
        j_idx = np.where(self.j_ids == j)[0]

        # If either ID not found, return nan
        if len(i_idx) == 0 or len(j_idx) == 0:
            return np.nan # 'ID not found in cost matrix: {i}, {j}'

        if len(i_idx) > 1:
            raise DuplicateIDError(True, self.video_id, self.frame, self.i_ids[i_idx])
        if len(j_idx) > 1:
            raise DuplicateIDError(False, self.video_id, self.frame, self.j_ids[j_idx])

        return float(self.cost_matrix[i_idx[0], j_idx[0]])

    def is_equal(self, other: 'CostMatrixData', tol: float = 1e-8) -> bool:
        if not np.array_equal(self.i_ids, other.i_ids) or not np.array_equal(self.j_ids, other.j_ids) or not np.allclose(self.cost_matrix, other.cost_matrix, rtol=tol, atol=tol):
            return False
        return True

    def ref_id2idx(self, id: np.dtype[np.object_]) -> int:
        if self._ref_id2idx_map is None:
            self._ref_id2idx_map = {id: idx for idx, id in enumerate(self.i_ids)}
        return self._ref_id2idx_map[id]

    def comp_id2idx(self, id: np.dtype[np.object_]) -> int:
        if self._comp_id2idx_map is None:
            self._comp_id2idx_map = {id: idx for idx, id in enumerate(self.j_ids)}
        return self._comp_id2idx_map[id]

    def construct_id2idx_lookup(self) -> None:
        """Eagerly build the id→index maps in this process before fork/spawn.

        Called from the orchestrator so each worker inherits prebuilt maps
        instead of rebuilding them per chunk.
        """
        if self._ref_id2idx_map is None:
            self._ref_id2idx_map = {id: idx for idx, id in enumerate(self.i_ids)}
        if self._comp_id2idx_map is None:
            self._comp_id2idx_map = {id: idx for idx, id in enumerate(self.j_ids)}

    def construct_assignment(self):
        self.match_rows, self.match_cols = linear_sum_assignment(-self.cost_matrix)

        # Create mapping from GT ID to matched tracker ID
        self.ref2comp_idx_map = {self.match_rows[i]: self.match_cols[i] for i in range(len(self.match_rows))}
        a = self.i_ids[self.match_rows]
        b = self.j_ids[self.match_cols]
        self.ref2comp_id_map = {a[i]: b[i] for i in range(len(a))}

    def copy(self) -> 'CostMatrixData':
        """Returns a copy of the cost matrix.

        Returns:
            A new CostMatrixData instance with copied data
        """
        result = CostMatrixData(i_ids=self.i_ids.copy(), j_ids=self.j_ids.copy(), cost_matrix=self.cost_matrix.copy(), video_id=self.video_id, frame=self.frame)
        result._ref_id2idx_map = self._ref_id2idx_map.copy() if self._ref_id2idx_map is not None else None
        result._comp_id2idx_map = self._comp_id2idx_map.copy() if self._comp_id2idx_map is not None else None
        result.match_rows = self.match_rows.copy() if self.match_rows is not None else None
        result.match_cols = self.match_cols.copy() if self.match_cols is not None else None
        result.ref2comp_idx_map = self.ref2comp_idx_map.copy() if self.ref2comp_idx_map is not None else None
        result.ref2comp_id_map = self.ref2comp_id_map.copy() if self.ref2comp_id_map is not None else None
        return result



class CostMatrixDataFrame (CostMatrixData):
    i_hashes: np.ndarray  # Array of reference hashes for the y axis of the cost matrix
    j_hashes: np.ndarray  # Array of comparison hashes for the x axis of the cost matrix


    def __init__(self, i_ids: np.ndarray, j_ids: np.ndarray, i_hashes: np.ndarray, j_hashes: np.ndarray, cost_matrix: np.ndarray, video_id: str, frame: int):
        super().__init__(i_ids, j_ids, cost_matrix, video_id, frame)
        self.i_hashes = i_hashes
        self.j_hashes = j_hashes

    def copy(self) -> 'CostMatrixDataFrame':
        """Returns a copy of the cost matrix.

        Returns:
            A new CostMatrixDataFrame instance with copied data
        """
        i_hashes = self.i_hashes.copy() if self.i_hashes is not None else None
        j_hashes = self.j_hashes.copy() if self.j_hashes is not None else None
        result = CostMatrixDataFrame(i_ids=self.i_ids.copy(), j_ids=self.j_ids.copy(), i_hashes=i_hashes, j_hashes=j_hashes, cost_matrix=self.cost_matrix.copy(), video_id=self.video_id, frame=self.frame)
        result._ref_id2idx_map = self._ref_id2idx_map.copy() if self._ref_id2idx_map is not None else None
        result._comp_id2idx_map = self._comp_id2idx_map.copy() if self._comp_id2idx_map is not None else None
        result.match_rows = self.match_rows.copy() if self.match_rows is not None else None
        result.match_cols = self.match_cols.copy() if self.match_cols is not None else None
        result.ref2comp_idx_map = self.ref2comp_idx_map.copy() if self.ref2comp_idx_map is not None else None
        result.ref2comp_id_map = self.ref2comp_id_map.copy() if self.ref2comp_id_map is not None else None
        return result
