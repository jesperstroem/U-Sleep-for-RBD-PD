import mne
from csdp_datastore import ABC

path = "C:/Users/au588953/abc"

d = ABC(path, path, max_num_subjects=1, filter=True)

d.port_data()

# data = mne.io.read_raw_edf("C:/Users/au588953/abc/polysomnography/edfs/baseline/abc-baseline-900001.edf", preload=True, verbose=False)

# data = mne.set_bipolar_reference(data, anode="M1", cathode="M2", ch_name="M1-M2", drop_refs=False)

# data.pick(picks=["M1", "M2", "M1-M2"]).plot()