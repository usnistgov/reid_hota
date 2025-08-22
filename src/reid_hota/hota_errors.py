

class HOTAConfigError(ValueError):
    """Base class for HOTA configuration errors."""
    pass

class HOTARuntimeError(ValueError):
    """Base class for HOTA runtime errors."""
    pass



class InvalidIDAlignmentMethodError(HOTAConfigError):
    """Raised when an invalid ID alignment method is provided."""
    
    def __init__(self, method: str, valid_methods: list):
        self.method = method
        self.valid_methods = valid_methods
        message = f"Invalid ID alignment method '{method}'. Must be one of: {valid_methods}"
        super().__init__(message)


class InvalidSimilarityMetricError(HOTAConfigError):
    """Raised when an invalid similarity metric is provided."""
    
    def __init__(self, metric: str, valid_metrics: list = None):
        self.metric = metric
        self.valid_metrics = valid_metrics
        if valid_metrics is None:
            message = f"Invalid similarity metric '{metric}'. "
        else:
            message = f"Invalid similarity metric '{metric}'. Must be one of: {valid_metrics}"
        super().__init__(message)


class EmptyIOUThresholdsError(HOTAConfigError):
    """Raised when IOU thresholds list is empty."""
    
    def __init__(self):
        message = "IOU thresholds list cannot be empty"
        super().__init__(message)


class UnsupportedBoxFormatError(HOTAConfigError):
    """Raised when an unsupported box format is provided."""
    
    def __init__(self, box_format: str):
        self.box_format = box_format
        
        message = f'Unsupported box format: {box_format}'
        super().__init__(message)


class InvalidIOUThresholdsRangeError(HOTAConfigError):
    """Raised when IOU thresholds are outside the valid range [0, 1]."""
    
    def __init__(self, thresholds):
        self.thresholds = thresholds
        message = f"IOU thresholds {thresholds} must all be in range [0, 1]"
        super().__init__(message)


class NonFiniteSimilarityValueError(HOTARuntimeError):
    """Raised when non-finite values are found in similarity matrix during HOTA computation."""
    
    def __init__(self, video_id: str, frame: int, i_ids, j_ids, cost_matrix):
        self.video_id = video_id
        self.frame = frame
        self.i_ids = i_ids
        self.j_ids = j_ids
        self.cost_matrix = cost_matrix
        
        message = (
            f"Non-finite value in matched_similarity_vals for video {video_id} frame {frame}\n"
            f"  sim_cost_matrix.i_ids: {i_ids}\n"
            f"  sim_cost_matrix.j_ids: {j_ids}\n"
            f"  sim_cost_matrix.cost_matrix: {cost_matrix}"
        )
        super().__init__(message)


class MissingVideoIDError(HOTARuntimeError):
    """Raised when video_id is None or missing during HOTA result merging."""
    
    def __init__(self, context: str = ""):
        self.context = context
        message = f"Missing Video Id found during HOTA data merge. video_id is None{f' in {context}' if context else ''}"
        super().__init__(message)


class DuplicateIDError(HOTARuntimeError):
    """Raised when duplicate IDs are found in a frame during HOTA computation."""
    
    def __init__(self, is_ground_truth: bool, video_id: str, frame: int, duplicate_ids):
        self.is_ground_truth = is_ground_truth
        self.video_id = video_id
        self.frame = frame
        self.duplicate_ids = duplicate_ids

        source_txt = "Ground Truth" if is_ground_truth else "Tracker"
        
        message = f'Duplice {source_txt} global ids found in video {video_id} at frame {frame}: {duplicate_ids}. reid_hota cannot disambiguate.'
        super().__init__(message)

