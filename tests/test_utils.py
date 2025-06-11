import json
import numpy as np


def validate_results(global_hota_data, gt_filepath, tol=1e-4):
    failed_keys = []
    gt_results = load_hota_results(gt_filepath)

    # don't validate counts type items
    keys_to_validate = ['TP','FP','FN','LocA','HOTA','AssA', 'AssRe', 'AssPr', 'DetA', 'DetRe', 'DetPr', 'OWTA', 'IDF1']
    
    for key in keys_to_validate:
        if not np.allclose(global_hota_data[key], gt_results[key], atol=tol):
            print(f"Failed on key: {key}")
            print(f"  difference: {global_hota_data[key] - gt_results[key]}")
            failed_keys.append(key)
    
    if len(failed_keys) > 0:
        raise AssertionError(f"HOTA test failed on keys: {failed_keys}")
    else:
        print("HOTA test passed")

def save_hota_results(combined_hota, fp):
    # Create a serializable dictionary from the combined_hota data
    hota_data_dict = combined_hota.get_dict()
    keys_to_delete = []
    
    for k,v in hota_data_dict.items():
        if 'counts' in k or 'hashes' in k:  
            keys_to_delete.append(k)
            continue
        if isinstance(v, np.ndarray):
            hota_data_dict[k] = v.tolist()
        elif isinstance(v, set):
            hota_data_dict[k] = list(v)
        elif isinstance(v, dict):
            hota_data_dict[k] = v
    
    for k in keys_to_delete:
        del hota_data_dict[k]

    # Save to JSON file
    assert fp.endswith('.json'), f"File must have a .json extension, got {fp}"
    with open(fp, 'w') as f:
        json.dump(hota_data_dict, f, indent=2)

def load_hota_results(fp):
    assert fp.endswith('.json'), f"File must have a .json extension, got {fp}"
    with open(fp, 'r') as f:
        hota_data_dict = json.load(f)
    return hota_data_dict


