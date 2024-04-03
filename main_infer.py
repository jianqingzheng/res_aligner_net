import argparse
import os
import numpy as np

from infer_tfkeras import infer
import ran_func as func

# name = "unpaired_ct_lung"
name = "unpaired_ct_abdomen"

parser = argparse.ArgumentParser()

#=======================================================================================================================
parser.add_argument(
        "--model_name",
        "-mn",
        help="network for training."
        "-mn for ",
        type=str,
        default="RAN4",
        required=False,
    )
parser.add_argument(
        "--data_name",
        "-dn",
        help="data name for training."
        "-dn for ",
        type=str,
        default=name,
        required=False,
    )

args = parser.parse_args()
#=======================================================================================================================

data_name=args.data_name
model_name=args.model_name
print(model_name)

rescale_factor=1
rescale_factor_label=1

int_range=[-100,300]

crop_sz = np.array([0, 0, 0])
print(crop_sz)

net_core = func.networks.get_net(model_name)


# use_lab=True
use_lab=False

model_path=os.path.join('.','models',data_name,data_name+'-'+model_name,'model')
data_path=os.path.join('.','data',data_name,'dataset')

print(os.path.abspath(data_path))

test_paths = os.path.join(data_path,'test_proc')

#=======================================================================================================================

infer(net_core=net_core,model_path=model_path,crop_sz=crop_sz,pair_type="unpaired",rescale_factor=rescale_factor,rescale_factor_label=rescale_factor_label,use_lab=use_lab,test_path=test_paths,model_name=model_name,int_range=int_range)
