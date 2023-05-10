import numpy as np
import yaml
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from utils import parser_from_dict
from algo import DP, DRO


if __name__ == "__main__":
    f = open("./params.yaml", encoding="utf-8")
    params = yaml.load(stream=f, Loader=yaml.FullLoader)
    args = parser_from_dict(params)

    # Run our baseline
    algo = DP(args)
    # policy = algo.run()
    # np.savetxt('./learnedpolicy/DP.csv', policy, delimiter=',')
    DP_policy = np.loadtxt('./learnedpolicy/DP.csv', delimiter=',')
    DP_total_rewards = algo.demo(DP_policy)
    print(DP_total_rewards)

    # Start Run DRO
    base_buffersize = 1000
    DRO_total_rewards = []

    # Exp 1: fix gamma, adjust buffer size
    for i in range(args.num):
        args.buffersize = 1000 * (i + 1)
        exp_name = 'DRO' + str(i) + '_' + str(args.buffersize)
        algo = DRO(args)
        DRO_policy = algo.run()
        DRO_rewards = algo.demo(DRO_policy)
        DRO_total_rewards.append(DRO_rewards)
        np.savetxt('./learnedpolicy/' + exp_name + '.csv', DRO_policy, delimiter=',')

    Reward_diff = np.array(DRO_total_rewards) - DP_total_rewards
    np.savetxt('./learnedpolicy/DP.csv')
    sns.set(rc={'figure.figsize': (16, 12)})
    sns.set_theme(style="whitegrid", font_scale=3)
    dict = {'Dataset Size': 1000*(np.arange(args.num)+1),
            'Rewards Difference': Reward_diff}
    pd = pd.DataFrame(dict)
    sns.lmplot(data=pd, x='Dataset Size', y='Rewards Difference')
    plt.show()

    # Exp 2: fix buffer size, adjust gamma
