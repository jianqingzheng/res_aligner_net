import argparse
import os
import numpy as np

import ran_func as func
from train_tfkeras import train

# name = "unpaired_ct_lung"
name = "unpaired_ct_abdomen"

#=======================================================================================================================
parser = argparse.ArgumentParser()

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
parser.add_argument(
    "--type",
    help="Training type, unpaired or paired or grouped or synthetic",
    default="",
    type=str,
    required=False,
)

parser.add_argument(
    "--max_epochs",
    help="The maximum number of epochs, -1 means following configuration.",
    type=int,
    default=0,
)
parser.add_argument(
    "--start_stage",
    help="The starting stage.",
    type=int,
    default=1,
    # default=2,
)

args = parser.parse_args()
#=======================================================================================================================


nb_epochs = [50, 100]
rescale_factor=1
restore_pre_train=False

if args.max_epochs > 0 and args.max_epochs is not None:
    restore_pre_train = True
    nb_epochs[args.start_stage-1] = args.max_epochs

if args.start_stage is 1:
    train_stages=[1,2]

elif args.start_stage is 2:
    train_stages=[2]


model_name=args.model_name
data_name=args.data_name

pair_type = "unpaired" if "unpaired" in data_name else "paired"

# network setting
print("model:"+model_name)
net_core = func.networks.get_net(model_name)


if data_name in ["unpaired_ct_lung"]:
    train_folder_name = 'train_proc'
    crop_sz = np.array([12, 12, 13])
    grad_weight = .05
    thresh = .01
    batch_size = 3
    num_lab=1
elif data_name in ["unpaired_ct_abdomen"]:
    train_folder_name = 'train_proc'
    crop_sz = np.array([12, 10, 20])
    grad_weight=.5
    thresh=.1
    batch_size = 3
    # batch_size = 1
    num_lab=9

rescale_factor=1
int_range=[-100,300]

model_path=os.path.join('.','models',data_name,data_name+'-'+model_name,'model')
data_path=os.path.join('.','data',data_name,'dataset')

print(os.path.abspath(data_path))


train_paths = os.path.join(data_path,train_folder_name)
print(train_paths)


#=======================================================================================================================

train(net_core=net_core, model_path=model_path, crop_sz=crop_sz, pair_type=pair_type, rescale_factor=rescale_factor,num_lab=num_lab,
      restore_pre_train=restore_pre_train, train_path=train_paths, batch_size=batch_size, nb_epochs=nb_epochs,
      train_stages=train_stages,grad_weight=grad_weight,thresh=thresh,int_range=int_range)
