import h5py
from csdp_datastore import ABC
from csdp_datastore.models import ChannelCalculations, Rereference, Mapping, TTRef

path = "C:/Users/au588953/abc"

reref = Rereference(first=Mapping(TTRef.LPA, TTRef.Fpz),
                    second=Mapping(TTRef.RPA, TTRef.Fpz),
                    result=Mapping(TTRef.LPA, TTRef.RPA))

config = ChannelCalculations(rereferences=[reref],
                             drop_existing=False)

d = ABC(path, path, max_num_subjects=1, filter=True, calculated_channel_config=config)

d.port_data()

# f = h5py.File("C:/Users/au588953/abc/abc.hdf5", "r")
# print(f["data"]["900001"]["baseline"]["psg"].keys())

# data = mne.io.read_raw_edf("C:/Users/au588953/abc/polysomnography/edfs/baseline/abc-baseline-900001.edf", preload=True, verbose=False)

# data = mne.set_bipolar_reference(data, anode="M1", cathode="M2", ch_name="M1-M2", drop_refs=False)

# data.pick(picks=["M1", "M2", "M1-M2"]).plot()