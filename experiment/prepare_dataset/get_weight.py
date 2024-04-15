from dgl.dev import *
from preprocess.freq import *
from utils import *

if __name__ == "__main__":
    
    args = get_args()
    
    print(f"{args=}")
    graph_name = str(args.graph_name)
    data_dir = args.data_dir
    batch_size = args.batch_size
    fanouts = args.fanouts.split(',')
    num_epoch = args.num_epoch
    for idx, fanout in enumerate(fanouts):
        fanouts[idx] = int(fanout)
    dir_path = os.path.dirname(os.path.realpath(__file__))
    log_path = os.path.join(dir_path, "logs/exp.csv")
    cfg = Config(graph_name=graph_name,
                       world_size=4,
                       num_epoch=num_epoch,
                       fanouts=fanouts,
                       batch_size=batch_size,
                       system="dgl-sample",
                       model="none",
                       cache_size="0GB",
                       hid_size=256,
                       log_path=log_path,
                       data_dir=data_dir)
           
    freq(cfg)