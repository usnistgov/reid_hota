import numpy as np
from dataclasses import dataclass, field



@dataclass
class Sparse2DMatrix:
    data_store: dict[tuple[np.dtype[np.object_], np.dtype[np.object_]], float] = field(default_factory=dict)

    def add_at(self, i: np.dtype[np.object_], j: np.dtype[np.object_], v: float) -> None:
        """Add a value to the element at position (i, j)."""
        self[i, j] = self[i, j] + v

    def get(self, i: np.dtype[np.object_], j: np.dtype[np.object_]) -> float:
        """Get the value at position (i, j)."""
        return self[i, j]

    def __getitem__(self, key) -> float:
        """Returns the value at position (i, j).
        
        Args:
            key: Either a tuple (i, j) or comma-separated indices i, j
            
        Returns:
            The value at position (i, j), or 0 if not found.
        """
        if isinstance(key, tuple) and len(key) == 2:
            i, j = key
        else:
            raise KeyError("Sparse2DMatrix indices must be a tuple of length 2")
        return self.data_store.get((i, j), 0)
    
    def __setitem__(self, key, value: float) -> None:
        """Sets the value at position (i, j).
        
        Args:
            key: Either a tuple (i, j) or comma-separated indices i, j
            value: The value to set.
        """
        if isinstance(key, tuple) and len(key) == 2:
            i, j = key
        else:
            raise KeyError("Sparse2DMatrix indices must be a tuple of length 2")
            
        if value == 0:
            # Remove zero values to maintain sparsity
            self.data_store.pop((i, j), None)
        else:
            self.data_store[(i, j)] = value
    
    def __delitem__(self, key) -> None:
        """Deletes the value at position (i, j).
        
        Args:
            key: Either a tuple (i, j) or comma-separated indices i, j
        """
        if isinstance(key, tuple) and len(key) == 2:
            i, j = key
        else:
            raise KeyError("Sparse2DMatrix indices must be a tuple of length 2")
            
        if (i, j) in self.data_store:
            del self.data_store[(i, j)]
    
    def __contains__(self, key) -> bool:
        """Checks if position (i, j) exists in the matrix.
        
        Args:
            key: Either a tuple (i, j) or comma-separated indices i, j
            
        Returns:
            True if the position exists (and has non-zero value), False otherwise.
        """
        if isinstance(key, tuple) and len(key) == 2:
            i, j = key
        else:
            return False
        return (i, j) in self.data_store
    
    def __len__(self) -> int:
        """Returns the number of non-zero elements.
        
        Returns:
            Number of non-zero elements in the matrix.
        """
        return len(self.data_store)
    
    def __iter__(self):
        """Allows iteration over the position tuples with non-zero values."""
        return iter(self.data_store.keys())
    
    def keys(self):
        """Returns the keys (position tuples) of non-zero elements."""
        return self.data_store.keys()
    
    def values(self):
        """Returns the values of non-zero elements."""
        return self.data_store.values()

    def items(self) -> list[tuple[tuple[np.dtype[np.object_], np.dtype[np.object_]], float]]:
        """Returns a list of (key, value) pairs, mimicking dict.items().
        
        Returns:
            List of tuples containing (key, value) pairs.
        """
        return list(self.data_store.items())
    
    def __iadd__(self, other: 'Sparse2DMatrix') -> 'Sparse2DMatrix':
        """Adds another Sparse2DMatrix to this one in-place.
        
        Args:
            other: Another Sparse2DMatrix to add to this one.
            
        Returns:
            Self, with values from other added.
        """
        # Add values from the other matrix to this one
        for (i, j), v in other.items():
            self.add_at(i, j, v)
        
        return self
    
    def __add__(self, other: 'Sparse2DMatrix') -> 'Sparse2DMatrix':
        """Adds another Sparse2DMatrix to this one, returning a new matrix.
        
        Args:
            other: Another Sparse2DMatrix to add to this one.
            
        Returns:
            New Sparse2DMatrix with values from both matrices added.
        """
        result = Sparse2DMatrix()
        result.data_store = self.data_store.copy()
        result += other
        return result
    
    def __repr__(self) -> str:
        """String representation of the matrix."""
        if not self.data_store:
            return "Sparse2DMatrix({})"
        items_str = ", ".join(f"{k}: {v}" for k, v in sorted(self.data_store.items()))
        return f"Sparse2DMatrix({{{items_str}}})"
    
    def clear(self) -> None:
        """Removes all elements from the matrix."""
        self.data_store.clear()
    
    def copy(self) -> 'Sparse2DMatrix':
        """Returns a copy of the matrix."""
        result = Sparse2DMatrix()
        result.data_store = self.data_store.copy()
        return result


@dataclass
class Sparse1DMatrix:
    data_store: dict[np.dtype[np.object_], float] = field(default_factory=dict)

    def add_at(self, i: np.dtype[np.object_], v: float) -> None:
        """Add a value to the element at index i."""
        self[i] = self[i] + v

    def get(self, i: np.dtype[np.object_]) -> float:
        """Get the value at index i."""
        return self[i]
    
    def __getitem__(self, i: np.dtype[np.object_]) -> float:
        """Returns the value at index i.
        
        Args:
            i: The index to get the value for.
            
        Returns:
            The value at index i, or 0 if not found.
        """
        return self.data_store.get(i, 0)
    
    def __setitem__(self, i: np.dtype[np.object_], value: float) -> None:
        """Sets the value at index i.
        
        Args:
            i: The index to set the value for.
            value: The value to set.
        """
        if value == 0:
            # Remove zero values to maintain sparsity
            self.data_store.pop(i, None)
        else:
            self.data_store[i] = value
    
    def __delitem__(self, i: np.dtype[np.object_]) -> None:
        """Deletes the value at index i.
        
        Args:
            i: The index to delete.
        """
        if i in self.data_store:
            del self.data_store[i]
    
    def __contains__(self, i: np.dtype[np.object_]) -> bool:
        """Checks if index i exists in the matrix.
        
        Args:
            i: The index to check for.
            
        Returns:
            True if the index exists (and has non-zero value), False otherwise.
        """
        return i in self.data_store
    
    def __len__(self) -> int:
        """Returns the number of non-zero elements.
        
        Returns:
            Number of non-zero elements in the matrix.
        """
        return len(self.data_store)
    
    def __iter__(self):
        """Allows iteration over the indices with non-zero values."""
        return iter(self.data_store.keys())
    
    def keys(self):
        """Returns the keys (indices) of non-zero elements."""
        return self.data_store.keys()
    
    def values(self):
        """Returns the values of non-zero elements."""
        return self.data_store.values()
    
    def items(self) -> list[tuple[np.dtype[np.object_], float]]:
        """Returns a list of (key, value) pairs, mimicking dict.items().
        
        Returns:
            List of tuples containing (key, value) pairs.
        """
        return list(self.data_store.items())
    
    def __iadd__(self, other: 'Sparse1DMatrix') -> 'Sparse1DMatrix':
        """Adds another Sparse1DMatrix to this one in-place.
        
        Args:
            other: Another Sparse1DMatrix to add to this one.
            
        Returns:
            Self, with values from other added.
        """
        # Add values from the other matrix to this one
        for i, v in other.items():
            self.add_at(i, v)
        
        return self
    
    def __add__(self, other: 'Sparse1DMatrix') -> 'Sparse1DMatrix':
        """Adds another Sparse1DMatrix to this one, returning a new matrix.
        
        Args:
            other: Another Sparse1DMatrix to add to this one.
            
        Returns:
            New Sparse1DMatrix with values from both matrices added.
        """
        result = Sparse1DMatrix()
        result.data_store = self.data_store.copy()
        result += other
        return result
    
    def __repr__(self) -> str:
        """String representation of the matrix."""
        if not self.data_store:
            return "Sparse1DMatrix({})"
        items_str = ", ".join(f"{k}: {v}" for k, v in sorted(self.data_store.items()))
        return f"Sparse1DMatrix({{{items_str}}})"
    
    def clear(self) -> None:
        """Removes all elements from the matrix."""
        self.data_store.clear()
    
    def copy(self) -> 'Sparse1DMatrix':
        """Returns a copy of the matrix."""
        result = Sparse1DMatrix()
        result.data_store = self.data_store.copy()
        return result