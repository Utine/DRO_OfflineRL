import numpy as np
import yaml
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import wandb
from utils import parser_from_dict
from algo import DP, DRO


if __name__ == "__main__":
    f = open("./params.yaml", encoding="utf-8")
    params = yaml.load(stream=f, Loader=yaml.FullLoader)
    args = parser_from_dict(params)

    wandb.init(project='OfflineRL', name='DRO')

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
    DRO_step_for_terminated = []

    # Exp 1: fix gamma, adjust buffer size
    for i in range(args.num):
        args.buffersize = 1000 * (i + 1)
        exp_name = 'DRO' + str(i) + '_' + str(args.buffersize)
        algo = DRO(args)
        DRO_policy = algo.run()
        DRO_rewards, step = algo.demo(DRO_policy)
        DRO_total_rewards.append(DRO_rewards)
        DRO_step_for_terminated.append(step)
        np.savetxt('./learnedpolicy/' + exp_name + '.csv', DRO_policy, delimiter=',')

    Reward_diff = np.array(DRO_total_rewards) - DP_total_rewards
    StepCost = np.array(DRO_step_for_terminated)
    np.savetxt('./csvs/reward_diff_with_datasize.csv', Reward_diff, delimiter=',')
    np.savetxt('./csvs/step_terminate_with_datasize.csv', StepCost, delimiter=',')
    sns.set(rc={'figure.figsize': (16, 12)})
    sns.set_theme(style="whitegrid", font_scale=3)
    dict = {'Dataset Size': 1000*(np.arange(args.num)+1),
            'Rewards Difference': Reward_diff,
            'Step Cost': StepCost}
    pd = pd.DataFrame(dict)
    sns.lmplot(data=pd, x='Dataset Size', y='Rewards Difference')
    plt.savefig('imgs/' + 'RewardDiff_Dataset_Size.png', dpi=200)
    wandb.log({"RewardDiff_Dataset_Size": wandb.Image('imgs/' + 'RewardDiff_Dataset_Size.png')})
    plt.show()
    plt.close()

    sns.lmplot(data=pd, x='Dataset Size', y='Step Cost')
    plt.savefig('imgs/' + 'StepCost_Dataset_Size.png', dpi=200)
    wandb.log({"StepCost_Dataset_Size": wandb.Image('imgs/' + 'StepCost_Dataset_Size.png')})
    plt.show()
    plt.close()

    # Exp 2: fix buffer size, adjust gamma
