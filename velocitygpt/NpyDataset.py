import numpy as np
from torch.utils.data import Dataset
import json
from typing import Dict, Any, Iterator, List, Tuple, Union, _TypingEllipsis
import torch
import copy
import os
import warnings as Warning

class NpyDataset(Dataset):
    def __init__(self,
                 paths: List[dict] = [],
                 dt_transformations: Union[List[callable], callable] = [], 
                 lb_transformations: Union[List[callable], callable] = [],
                 dtype: np.dtype = np.float32,
                 ltype: np.dtype = np.float32, 
                 norm: int = 0,
                 stride: int = 1,
                 mode: str = 'slice', 
                 line_mode: str = 'both',
                 window_w: int = 128,
                 window_h: int = 128,
                 stride_w: int = 30,
                 stride_h: int = 30
                 ):
        """
        Initialize the NpyDataset.

        Args:
            paths (List[Dict[str, Any]]): A list of dictionaries containing the paths to the seismic data and labels.
                Each dictionary should have the following structure:
                {
                    'data': str,  # path to data file
                    'label': str,  # path to label file (optional) - if not provided, labels will be zeros (can be ignored/used in testing)
                    'order': Tuple[str, str, str],  # e.g., ('x', 'y', 'z') 
                    'range': [Dict[str, Tuple[float, float]]]  # e.g., {'x': (0, 1), 'y': (0, 1), 'z': (0, 1)} (optional: any missing dimension will default to full range)
                }
            dt_transformations (Optional[Union[List[Callable], Callable]]): Transformations to apply to the data.
            lb_transformations (Optional[Union[List[Callable], Callable]]): Transformations to apply to the labels.
            dtype (np.dtype): The datatype to use for the data.
            ltype (np.dtype): The datatype to use for the labels.
            norm (int): Whether to normalize the data when loading. (0 = no normalization, 1 = normalize before applying transformations, 2 = normalize after applying transformations)
            stride (int): The stride to use when creating slices or windows (default: 1). a.k.a. steps or jumps when indexing the data.
            mode (str): The mode to use for the dataset. Options: 'windowed', 'slice', 'traces'.
            line_mode (str): The line mode to use for slice and windowed modes. Options: 'both', 'iline', 'xline'.
            window_w (int): The width of the windowed slice (only used in 'windowed' mode).
            window_h (int): The height of the windowed slice (only used in 'windowed' mode).
            stride_w (int): The stride to use when creating windowed slices on width (only used in 'windowed' mode).
            stride_h (int): The stride to use when creating windowed slices on height (only used in 'windowed' mode).
        """
        
        self.mode, self.line_mode = self._validate_mode(mode, line_mode)
        self.paths = self._validate_paths(paths)
        self.dtype = np.dtype(dtype)
        self.ltype = np.dtype(ltype)
        self.norm = norm
        self.stride = stride
        
        self.dt_t = self._setup_transformations(dt_transformations)
        self.lb_t = self._setup_transformations(lb_transformations)
        
        if not any([t.__name__ == 'normalize_item' for t in self.dt_t]): 
            if self.norm == 1:
                self.dt_t.insert(0, normalize_item)
            elif self.norm == 2:
                self.dt_t.append(normalize_item)
        
        self.window_params = {
            'window_w': window_w,
            'window_h': window_h,
            'stride_w': stride_w,
            'stride_h': stride_h
        }
        
        self.raw_data, self.raw_labels, self.orders, self.ranges, self.n_items_per_file, self.window_sizes = self._initialize_data() 
    
    def _setup_transformations(self, transformations: Any) -> List[callable]:
        """Setup the transformations."""
        t = []
        
        if isinstance(transformations, list):
            t = copy.deepcopy(transformations)
        else:
            t = [copy.deepcopy(transformations)]
        
        return t
               
    def _validate_paths(self, paths: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Validate the paths and their structure."""
        if not paths:
            raise ValueError("No paths provided.")
        for path in paths:
            if 'data' not in path:
                raise ValueError(f"Missing 'data' key in path: {path}")
            if 'order' not in path:
                raise ValueError(f"Missing 'order' key in path: {path}")
            if len(path['order']) != 3:
                raise ValueError(f"Invalid 'order' in path: {path}. Must be a tuple/List of 3 strings.")
            
            if isinstance(path['order'], List):
                path['order'] = tuple(path['order'])
            
        return copy.deepcopy(paths)
    
    def _validate_mode(self, mode: str, line_mode) -> str:
        """Validate the mode."""
        valid_modes = ['slice', 'windowed', 'traces']
        valid_line = ['iline', 'xline', 'both']
        if mode not in valid_modes:
            raise ValueError(f"Invalid mode: {mode}. Must be one of {valid_modes}")
        if line_mode not in valid_line:
            raise ValueError(f"Invalid line mode: {line_mode}. Must be one of {valid_line}")
        
        return mode, line_mode
        
    def _initialize_data(self) -> Tuple[List[np.ndarray], List[np.ndarray], List[Tuple[str, str, str]], List[int]]:
        data = []
        labels = []
        orders = []
        ranges = []
        n_items = []
        window_sizes = []
        
        for path in self.paths:
            _dt = np.load(path['data'], mmap_mode='r')
            
            if 'label' in path:
                _lb = np.load(path['label'], mmap_mode='r')
            else: 
                _lb = np.zeros_like(_dt)
                
            _or = path['order']
            x_pos = _or.index('x')
            y_pos = _or.index('y')
            
            x_st = 0
            x_end = _dt.shape[x_pos]
            
            y_st = 0
            y_end = _dt.shape[y_pos]
            
            z_st = 0
            z_end = _dt.shape[_or.index('z')]
            
            if path.get('range') is not None:
                if 'x' in path ['range']:
                    x_range = path['range']['x']
                    x_st = (int) (x_range[0] * _dt.shape[x_pos])
                    x_end = (int) (x_range[1] * _dt.shape[x_pos])
                else: 
                    msg = (f"No range specified for x in path {path}, defaulting to full range")
                    Warning.warn(msg)
                
                if 'y' in path ['range']:
                    y_range = path['range']['y']
                    y_st = (int) (y_range[0] * _dt.shape[y_pos])
                    y_end = (int) (y_range[1] * _dt.shape[y_pos])
                else:
                    msg = (f"No range specified for y in path {path}, defaulting to full range") 
                    Warning.warn(msg)
                
                if 'z' in path ['range']:    
                    z_range = path['range']['z']
                    z_st = (int) (z_range[0] * _dt.shape[_or.index('z')])
                    z_end = (int) (z_range[1] * _dt.shape[_or.index('z')])
                    _dt = _dt[z_st:z_end]
                    _lb = _lb[z_st:z_end]
                    msg = (f"Range specified for z in path: {path}, slicing the data and labels from (0, {_dt.shape[_or.index('z')]}) to ({z_st}, {z_end})")
                    Warning.warn(msg)
            
            data.append(_dt)
            labels.append(_lb)
            orders.append(path['order'])
            ranges.append((x_st, x_end, y_st, y_end))
            n_x = ((x_end - x_st) + (self.stride - 1))//self.stride
            n_y = ((y_end - y_st) + (self.stride - 1))//self.stride

            if self.mode == 'traces':                                
                n_items.append(n_x * n_y)
            elif self.mode == 'slice':
                if self.line_mode == "both":
                    n_items.append(n_x + n_y)
                elif self.line_mode == "iline":
                    n_items.append(n_x)
                else: 
                    n_items.append(n_y)
            else:
                # windows mode:
                ws = self._calculate_window_size(z_end - z_st, y_end - y_st, x_end - x_st)
                window_sizes.append(ws)
                if self.line_mode == "both":
                    n_i = ws['s1'][0] * ws['s1'][1] * (n_x) + ws['s2'][0] * ws['s2'][1] * (n_y)
                elif self.line_mode == "iline":
                    n_i = ws['s1'][0] * ws['s1'][1] * (n_x)
                else:
                    n_i = ws['s2'][0] * ws['s2'][1] * (n_y)
                    
                n_items.append(n_i)
            
        return data, labels, orders, ranges, n_items, window_sizes
    
    def __len__(self) -> int:
        return sum(self.n_items_per_file)
    
    def _get_shape (self, f_idx: int, dim: str) -> int:
        _dt = self.raw_data[f_idx]
        _or = self.orders[f_idx]
        _pos = _or.index(dim)
        if dim == 'z': 
            return _dt.shape[_pos]
        if dim == 'x': 
            x_st, x_end, _, _ = self.ranges[f_idx]
            n_x = ((x_end - x_st) + (self.stride - 1))//self.stride
            return n_x
        if dim == 'y':
            _, _, y_st, y_end = self.ranges[f_idx]
            n_y = ((y_end - y_st) + (self.stride - 1))//self.stride
            return n_y
        
        raise ValueError(f"Invalid dimension {dim}")
    
    def _calculate_window_size(self, z:int, wy:int, wx:int):
            ws = {}
            h = z
            # slice 1
            w = wy
            n_h = max(0, (h - self.window_params['window_h']) // self.window_params['stride_h'] + 1)
            n_w = max(0, (w - self.window_params['window_w']) // self.window_params['stride_w'] + 1)
            ws['s1'] = (n_h, n_w)
            
            # slice 2
            w = wx
            n_h = max(0, (h - self.window_params['window_h']) // self.window_params['stride_h'] + 1)
            n_w = max(0, (w - self.window_params['window_w']) // self.window_params['stride_w'] + 1)
            ws['s2'] = (n_h, n_w)
            
            return copy.deepcopy(ws) 
    
    def __getitem__(self, idx:int) -> Tuple[np.ndarray, np.ndarray]:
        if self.mode == 'slice': 
            return self._get_slice(idx)
        elif self.mode == 'windowed':
            return self._get_windowed_slice(idx)
        elif self.mode == 'traces':
            return self._get_trace(idx)
        else:
            raise ValueError(f"Invalid mode {self.mode}")       
   
    def _find_file (self, idx: int) -> int:
        _idx = idx
        _file_idx = 0
        while _idx >= self.n_items_per_file[_file_idx]:
            _idx -= self.n_items_per_file[_file_idx]
            _file_idx += 1
        return _file_idx
        
    def _get_trace(self, idx: int) -> Tuple[np.ndarray]:
        """
        Get the trace at the given index and its label
        """
        _file_idx = self._find_file(idx)
        
        # print ("in file", _file_idx, "index", _idx, "total items", self.n_items_per_file[_file_idx])
        
        _dt = self.raw_data[_file_idx]
        _lb = self.raw_labels[_file_idx]
        _rg = self.ranges[_file_idx]
        _or = self.orders[_file_idx]
        
        idx -= sum(self.n_items_per_file[:_file_idx])
        
        # print ("searching for which iline in: ", (_rg[3] , " to ",  _rg[2]))
        
        iline_offset = _rg[0]
        xline_offset = _rg[2]
        
        n_ilines = (_rg[3] - _rg[2] + self.stride - 1) // self.stride
        
        iline_idx = (idx // (n_ilines)) * self.stride + iline_offset
        
        # print ("getting iline at", iline_idx)
        
        iline_idx = (slice(None),  iline_idx, slice(None))

        # get the iline slice of the data
        _dt = get_transposed_slice(_dt, _or, ('z', 'x', 'y'), iline_idx).astype(self.dtype)
        _lb = get_transposed_slice(_lb, _or, ('z', 'x', 'y'), iline_idx).astype(self.ltype)
        # shape: (z, x)
        
        # find which trace to get
        trace_idx = (idx % n_ilines) * self.stride + xline_offset
        
        _dt = _dt[:, trace_idx]
        _lb = _lb[:, trace_idx]
        
        _dt, _lb = self._apply_transformations(_dt, _lb)
            
        return _dt, _lb
        
    def _get_slice(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the slice of seismic data and labels
        """
        # find the file that contains the index
        _file_idx = self._find_file(idx)
        
        _dt = self.raw_data[_file_idx]
        _lb = self.raw_labels[_file_idx]
        _or = self.orders[_file_idx]
        _rg = self.ranges[_file_idx]
        
        idx -= sum(self.n_items_per_file[:_file_idx])
        
        # find the 'x' index in _or
        n_ilines = self._get_shape(_file_idx, 'x')
        
        if self.line_mode == "both":
            # assuming the data is transposed to ('z', 'x', 'y') -- done on the fly later on 
            if idx < n_ilines:
                # indexing along the x axis
                idx *= self.stride # convert to the actual index
                idx += _rg[0] # add the starting index
                y_slice = slice(_rg[2], _rg[3]) # get the y slice
                idx = (slice(None), idx, y_slice)
            else:
                # indexing along the y axis
                idx -= n_ilines
                idx *= self.stride # convert to the actual index
                idx += _rg[2] # add the starting index
                x_slice = slice(_rg[0], _rg[1]) # get the x slice
                idx = (slice(None), x_slice, idx)
        elif self.line_mode == "iline":
            idx *= self.stride # convert to the actual index
            idx += _rg[0]
            y_slice = slice(_rg[2], _rg[3]) # get the y slice
            idx = (slice(None), idx, y_slice)
        else: 
            idx *= self.stride # convert to the actual index
            idx += _rg[2]
            x_slice = slice(_rg[0], _rg[1]) # get the x slice
            idx = (slice(None), x_slice, idx)
        
        # print (_file_idx, idx, x_pos)
        _dt = get_transposed_slice(_dt, _or, ('z', 'x', 'y'), idx).astype(self.dtype)
        _lb = get_transposed_slice(_lb, _or, ('z', 'x', 'y'), idx).astype(self.ltype)
        
        _dt, _lb = self._apply_transformations(_dt, _lb)
        
        return _dt, _lb
    
    def _get_windowed_slice(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        # find the file that contains the index
        _file_idx = self._find_file(idx)
        
        _idx = idx - sum(self.n_items_per_file[:_file_idx])
        
        _dt = self.raw_data[_file_idx]
        _lb = self.raw_labels[_file_idx]
        _or = self.orders[_file_idx]
        _rg = self.ranges[_file_idx]
        _ws = self.window_sizes[_file_idx]
        
        # find the slice that contains the index
        # s1 moves along the x axis first
        n_iline = self._get_shape(_file_idx, 'x')
        
        if self.line_mode == "both":
            if _idx < _ws['s1'][0] * _ws['s1'][1] * n_iline:
                # along the x axis
                slice_idx = _idx // (_ws['s1'][0] * _ws['s1'][1])
                
                slice_idx *= self.stride # convert to the actual index
                slice_idx += _rg[0] # add the starting index
                
                y_slice = slice(_rg[2], _rg[3]) # get the y slice
                slice_idx = (slice(None), slice_idx, y_slice)
                
                window_idx = _idx % (_ws['s1'][0] * _ws['s1'][1])
                # print ("getting an inline at ", slice_idx, "window at", window_idx)
                _ws = _ws['s1']
            else:
                # from a crossline
                slice_idx = (_idx - _ws['s1'][0] * _ws['s1'][1] * n_iline) // (_ws['s2'][0] * _ws['s2'][1])
                slice_idx *= self.stride # convert to the actual index
                slice_idx += _rg[2] # add the starting index
                
            
                x_slice = slice(_rg[0], _rg[1]) # get the x slice
                slice_idx = (slice(None), x_slice, slice_idx)
                
                
                window_idx = (_idx - _ws['s1'][0] * _ws['s1'][1] * n_iline) % (_ws['s2'][0] * _ws['s2'][1])
                # print ("getting a xline at ", slice_idx, "window at", window_idx)
                _ws = _ws['s2']
            
        elif self.line_mode == "iline":
            
            slice_idx = _idx // (_ws['s1'][0] * _ws['s1'][1])
            slice_idx *= self.stride # convert to the actual index
            slice_idx += _rg[0] # add the starting index
            
            y_slice = slice(_rg[2], _rg[3])
            slice_idx = (slice(None), slice_idx, y_slice)
            
            window_idx = _idx % (_ws['s1'][0] * _ws['s1'][1])
            _ws = _ws['s1']    
        else: 
            
            slice_idx = _idx // (_ws['s2'][0] * _ws['s2'][1])
            slice_idx *= self.stride # convert to the actual index
            slice_idx += _rg[2] # add the starting index
            
            x_slice = slice(_rg[0], _rg[1])
            slice_idx = (slice(None), x_slice, slice_idx)
            
            window_idx = _idx % (_ws['s2'][0] * _ws['s2'][1])
            _ws = _ws['s2']
        
        _dt = get_transposed_slice(_dt, _or, ('z', 'x', 'y'), slice_idx).astype(self.dtype)
        _lb = get_transposed_slice(_lb, _or, ('z', 'x', 'y'), slice_idx).astype(self.ltype)
        
        # get the window from the slice 
        _dt, _lb = self._get_window(_dt, _lb, window_idx, _ws)
        
        _dt, _lb = self._apply_transformations(_dt, _lb)

        return _dt, _lb
    
    def _get_window(self, dt: np.ndarray, lb: np.ndarray, idx: int, ws) -> Tuple[np.ndarray, np.ndarray]:
        # given a slice, get the window with the given index
        # ws : h x w
        row = idx // ws[1]
        col = idx % ws[1]
        
        st_h = row * self.window_params['stride_h']
        st_w = col * self.window_params['stride_w']
        
        # print ("window at", st_h, st_w, " with shape: ", dt.shape)
        dt = dt[st_h:st_h+self.window_params['window_h'], st_w:st_w+self.window_params['window_w']]
        lb = lb[st_h:st_h+self.window_params['window_h'], st_w:st_w+self.window_params['window_w']]
        return dt, lb
    
    def _apply_transformations(self, dt: np.ndarray, lb: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        for t in self.dt_t:
            dt = t(dt)
        
        for t in self.lb_t:
            lb = t(lb)
        
        return dt, lb
    
    def __del__(self):
        for file in self.raw_data:
            del file
        pass
                 
    def __iter__(self) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        for i in range(len(self)):
            yield self[i]
           
    def save_dataset(self,
                     path: str, 
                     chunk_size: int = 0):
        """
        Save the dataset - transformed to npy file
        - chunk_size: the number of slices to save in each file (0: save all in one file)
        - path: the path to save the dataset
        """
        raise NotImplementedError("Method not implemented")
        pass
    
    def load_dataset(self,
                     path: str):
        """
        Load the dataset from a given path
        """
        raise NotImplementedError("Method not implemented")
        pass
    
    def save_config(self, path: str) -> str:
        """
        Save the metadata of the dataset to a given path or a json file
        
        Returns:
        str: The path to the saved file
        """
        # check if the path is a directory or json file
        if path.endswith('.json'):
            with open(path, 'w') as f:
                mt = self.get_config()
                # print (mt)
                json.dump(mt, f)
                return path
        else: 
            # make sure the path exists and is a directory
            if not os.path.exists(path):
                os.makedirs(path)
            with open(os.path.join(path, 'metadata.json'), 'w') as f:
                json.dump(self.get_metadata(), f)              
            return os.path.join(path, 'metadata.json')
        
    def get_config(self) -> Dict[str, Any]:
        # compose all the metadata of the dataset to a dictionary
        config = {
            'paths': self.paths,
            'dt_transformations': [t.__name__ for t in self.dt_t],
            'lb_transformations': [t.__name__ for t in self.lb_t],
            'dtype': str(self.dtype),
            'ltype': str(self.ltype),
            'norm': self.norm,
            'mode': self.mode,
            'line_mode': self.line_mode,
            'window_w': self.window_params['window_w'],
            'window_h': self.window_params['window_h'],
            'stride_w': self.window_params['stride_w'],
            'stride_h': self.window_params['stride_h']
        }
        return config
    
    @classmethod
    def from_config_path(cls, path: str) -> 'NpyDataset':
        """
        Load the metadata of the dataset from a given path
        """
        with open(path, 'r') as f:
            config = json.load(f)
            return cls.from_config(config)
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'NpyDataset':
        config['dtype'] = np.dtype(config['dtype'])
        config['ltype'] = np.dtype(config['ltype'])
        
        print (os.getcwd())
        
        # Recreate transformation functions (assuming they are defined in the global scope)
        config['dt_transformations'] = [globals()[name] for name in config['dt_transformations']]
        config['lb_transformations'] = [globals()[name] for name in config['lb_transformations']]
        
        return cls(**config)
        
    def add_transforms(self, dt: List[callable] = [], lb: List[callable] = [], show_order: bool = False):
        """
        Add transformations to the dataset
        """
        for t in dt:
            self.dt_t.append(copy.deepcopy(t))
        for t in lb:
            self.lb_t.append(copy.deepcopy(t))
            
        if show_order:
            print ("Data Transformations:")
            for t in self.dt_t:
                print (t)
            print ("Label Transformations:")
            for t in self.lb_t:
                print (t)
                
    def __repr__(self) -> str:
        return f"NpyDataset: {len(self)} items, {self.get_config()}"
            
    @staticmethod
    def create_collate_fn(self, type = "window", **kwargs):
        if type == "window":
            return self.create_windowed_collate_fn(**kwargs)
        elif type == "padded":
            return self.create_padded_collate_fn(**kwargs)

    @staticmethod
    def create_windowed_collate_fn(self, norm_flag: bool = True, window_w: int = 128, window_h: int = 128, stride_w: int = 30, stride_h: int = 30):
        """
        Create a collate function that return windowed slices:
            if in windowed mode, the collate function would use a batch of windowed slices (windowed dataset) as specified by the class parameters, not the function, and return them 
             - Here you would get the same number of samples as the batch size
             - norm_falg: normalize each window in the batch
                - Note: if the dataset normalization flag is on then this would result in applying normalization twice on each window
            
            if in original mode, the collate function would use a batch of full slices, window them and return them
            - Here your batch is how many slices to use 
            - The number of samples returned would be the number of windows created using these slices
            - norm_falg: normalize each window in the batch
                - Note: if the dataset normalization flag is on then this would result in applying normalization twice (once on the slice and once on the window)
            
            if in traces mode, the collate function would use a batch of traces, window them and return them
            - Here your batch is how many traces to use
            
        """
    
        if self.mode == 'windowed':            
            def windowed_collate(batch):
                # Batch is already windowed, just need to stack and convert to tensors
                data = [item[0] for item in batch]
                labels = [item[1] for item in batch]
                
                # Convert to torch tensors and stack
                data = torch.stack([torch.from_numpy(d) for d in data])
                labels = torch.stack([torch.from_numpy(l) for l in labels])
                
                if norm_flag: 
                    print ("normalizing windows")
                    # normalize each item in the batch (a single window)
                    data = normalize_windows(data)
                return data, labels
            
            return copy.deepcopy(windowed_collate)
        
        elif self.mode == 'slice':
            def original_collate(batch):
                # print("Original collate")
                # Process each slice individually
                data_windows_list = []
                label_windows_list = []
                
                # one item at a time as the slices might be in different shapes
                for item in batch:
                    data = torch.from_numpy(item[0])
                    label = torch.from_numpy(item[1])
                    
                    # Window the data and labels
                    data_windows = data.unfold(0, window_h, stride_h).unfold(1, window_w, stride_w)
                    label_windows = label.unfold(0, window_h, stride_h).unfold(1, window_w, stride_w)
                    
                    # Reshape to have all windows as the first dimension
                    h, w, w_h, w_w = data_windows.shape
                    data_windows = data_windows.reshape(-1, w_h, w_w)
                    label_windows = label_windows.reshape(-1, w_h, w_w)
                    
                    # print (data_windows.shape, label_windows.shape)
                    
                    data_windows_list.append(data_windows)
                    label_windows_list.append(label_windows)
                
                data_windows = torch.cat(data_windows_list, dim=0)
                label_windows = torch.cat(label_windows_list, dim=0)
                
                if norm_flag: 
                    # print ("normalizing----------")
                    # normalize each item in the batch (a single window)
                    data_windows = normalize_windows(data_windows)
                
                return data_windows, label_windows
            
            return copy.deepcopy(original_collate)
            
        elif self.mode == 'traces':
            def traces_collate(batch):
                # Process each slice individually
                # window the traces by the given window_x 
                # window_y is redundant as it is always 1
                
                data_windows_list = []
                label_windows_list = []
                
                # one item at a time as the traces might be of different lengths
                for item in batch:
                    data = torch.from_numpy(item[0]) # shape: (z, )
                    label = torch.from_numpy(item[1]) # shape: (z, )
                    
                    # window over the z axis
                    data_windows = data.unfold(0, window_w, stride_w)
                    label_windows = label.unfold(0, window_w, stride_w)
                    
                    # Reshape to have all windows as the first dimension
                    h, window_h = data_windows.shape
                    
                    data_windows = data_windows.reshape(-1, window_h)
                    label_windows = label_windows.reshape(-1, window_h)
                    
                    
                    data_windows_list.append(data_windows)
                    label_windows_list.append(label_windows)
                
                data_windows = torch.cat(data_windows_list, dim=0)
                label_windows = torch.cat(label_windows_list, dim=0)
                
                if norm_flag:
                    # normalize each item in the batch (a single window)
                    data_windows = normalize_windows(data_windows)
                    
                return data_windows, label_windows
            
            return copy.deepcopy(traces_collate)
        
    @staticmethod 
    def create_padded_collate_fn(self, norm_flag: bool = True):
        """
        Create a collate function that return padded traces/slices
        """
        raise NotImplementedError("Method not implemented")
        
# helper functions
def get_transposed_slice(
        data: np.ndarray,
        current_order: Tuple[str, str, str],
        transposed_order: Tuple[str, str, str],
        transposed_index: Tuple[Union[int, slice], Union[int, slice], Union[int, slice]]
    ) -> np.ndarray:
        """
        Get a slice from the original 3D data as if it were transposed to a given order
        
        Args:
        data: The original 3D numpy array
        current_order: Tuple representing the current order of dimensions (e.g., ('x', 'y', 'z'))
        transposed_order: Tuple representing the desired order of dimensions (e.g., ('z', 'x', 'y'))
        transposed_index: The indexing in the transposed order, must be a 3-tuple with either integers or slices
        
        Returns:
        np.ndarray: The slice of the data as if it were transposed
        """
        dim_map = tuple(current_order.index(dim) for dim in transposed_order)
        
        original_index = [0, 0, 0]
        original_index[dim_map[0]] = transposed_index[0]
        original_index[dim_map[1]] = transposed_index[1]
        original_index[dim_map[2]] = transposed_index[2]
        
        result = data[tuple(original_index)]
        
        remaining_dims = [(i if i < len(result.shape) else len(result.shape) - 1) for i, idx in enumerate(transposed_index)]
        
        # account for the need to transpose the final result 
        if len(remaining_dims) > 1:
            transpose_map = {dim_map[i]: i for i in remaining_dims}
            transpose_order = tuple(transpose_map[i] for i in range(3) if i in transpose_map)
            result = np.transpose(result, transpose_order)
        
        return result
    
def normalize_windows(windows): 
    if windows.ndim == 3:
        # shape (batch, z, x)
        # Given a batch of windows, normalize each window individually
        mean = windows.mean(dim=(1, 2), keepdim=True)
        std = windows.std(dim=(1, 2), keepdim=True)
        # print ("Mean:", mean, "Std:", std)
        windows = (windows - mean) / (std + 1e-8)
    else: 
        # shape: (batch, z)
        mean = windows.mean(dim=1, keepdim=True)
        std = windows.std(dim=1, keepdim=True)
        windows = (windows - mean) / (std + 1e-8)
        
    return windows

def normalize_item(item):
    # Normalize a single slice
    mean = item.mean()
    std = item.std()
    item =(item - mean) / (std + 1e-8)
    # print ("Normalized:", slice.mean(), slice.std())
    return item


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.indices = list(range(len(dataset)))

    def __len__(self):
        # return len(self.indices)
        return len(self.indices)
    
    def _indices_to_labels(self, indices):
        labels = []
        for idx in indices:
            l = self.dataset[idx][1]
            labels.append(l)
        labels = torch.from_numpy(np.array(labels))
        return labels

    def __getitem__(self, index):
        d, l = self.dataset[index]
        return d, index