import numpy as np
import yaml
import matplotlib.pyplot as plt
from utils import parser_from_dict
from algo import DP, DRO


if __name__ == "__main__":
    f = open("./params.yaml", encoding="utf-8")
    params = yaml.load(stream=f, Loader=yaml.FullLoader)
    args = parser_from_dict(params)

    # algo = DP(args)
    # policy = algo.run()
    # np.savetxt('./learnedpolicy/DP.csv', policy, delimiter=',')

    algo = DRO(args)
    policy = algo.run()
    np.savetxt('./learnedpolicy/DRO.csv', policy, delimiter=',')