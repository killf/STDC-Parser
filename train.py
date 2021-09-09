from solver import Solver
from config import Config

if __name__ == '__main__':
    cfg = Config()
    # cfg.data_dir = "/data/face/parsing/dataset/ibugmask_release"
    cfg.model_args.backbone = "STDCNet1446"
    cfg.model_args.pretrain_model = "snapshot/STDCNet1446_76.47.tar"
    # cfg.do_val = False
    cfg.batch_size = 8

    solver = Solver(cfg)
    solver.train()
