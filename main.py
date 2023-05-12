import numpy as np
import yaml
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import wandb
from utils import parser_from_dict
from algo import DP, DRO, DP_EXP


if __name__ == "__main__":
    f = open("./params.yaml", encoding="utf-8")
    params = yaml.load(stream=f, Loader=yaml.FullLoader)
    args = parser_from_dict(params)

    wandb.init(project='OfflineRL', name='DRO')

    # Init: Run DP as baseline
    algo = DP(args)
    # policy = algo.run()
    # np.savetxt('./learnedpolicy/DP.csv', policy, delimiter=',')
    DP_policy = np.loadtxt('./learnedpolicy/DP.csv', delimiter=',')
    DP_total_rewards, DP_step_cost = algo.demo(DP_policy)
    print(DP_total_rewards, DP_step_cost)

    # Exp1: Run DRO

    # 1): fix gamma, adjust buffer size
    base_buffersize = 1000
    DRO_total_rewards = []
    DRO_step_for_terminated = []
    for i in range(args.datasize_num):
        args.buffersize = base_buffersize + i*500
        exp_name = 'DRO' + '_exp1_' + str(i) + '_' + str(args.buffersize)
        algo = DRO(args)
        DRO_policy = algo.run()
        DRO_rewards, step = algo.run_with_current_policy(DRO_policy)
        DRO_total_rewards.append(DRO_rewards)
        DRO_step_for_terminated.append(step)
        np.savetxt('./learnedpolicy/' + exp_name + '.csv', DRO_policy, delimiter=',')

    Reward_diff = np.array(DRO_total_rewards) - DP_total_rewards
    StepCost = np.array(DRO_step_for_terminated)
    np.savetxt('./csvs/DRO/reward_diff_with_datasize.csv', Reward_diff, delimiter=',')
    np.savetxt('./csvs/DRO/step_terminate_with_datasize.csv', StepCost, delimiter=',')
    # Reward_diff = np.loadtxt('./csvs/reward_diff_with_datasize.csv', delimiter=',')
    # StepCost = np.loadtxt('./csvs/step_terminate_with_datasize.csv', delimiter=',')
    sns.set(rc={'figure.figsize': (16, 12)})
    sns.set_theme(style="whitegrid")
    dict = {'Dataset Size': base_buffersize + np.arange(args.datasize_num)*500,
            'Rewards Difference': np.abs(Reward_diff),
            'Step Cost': StepCost}
    pd = pd.DataFrame(dict)
    sns.lmplot(data=pd, x='Dataset Size', y='Rewards Difference')
    plt.savefig('imgs/DRO/' + 'RewardDiff_Dataset_Size.png', dpi=200)
    wandb.log({"DRO_RewardDiff_Dataset_Size": wandb.Image('imgs/' + 'RewardDiff_Dataset_Size.png')})
    plt.show()
    plt.close()

    sns.lmplot(data=pd, x='Dataset Size', y='Step Cost')
    plt.savefig('imgs/DRO/' + 'StepCost_Dataset_Size.png', dpi=200)
    wandb.log({"DRO_StepCost_Dataset_Size": wandb.Image('imgs/' + 'StepCost_Dataset_Size.png')})
    plt.show()
    plt.close()

    # 2): fix buffer size, adjust gamma
    base_gamma = 1
    DRO_total_rewards = []
    DRO_step_for_terminated = []

    for i in range(args.gamma_num):
        args.gamma = base_gamma - 0.02 * i
        exp_name = 'DRO' + '_exp2_' + str(i) + '_' + str(args.gamma)
        algo = DRO(args)
        DRO_policy, DRO_rewards, step = algo.run()
        DRO_total_rewards.append(DRO_rewards)
        DRO_step_for_terminated.append(step)
        np.savetxt('./learnedpolicy/' + exp_name + '.csv', DRO_policy, delimiter=',')

    Reward_diff = np.array(DRO_total_rewards) - DP_total_rewards
    StepCost = np.array(DRO_step_for_terminated)
    np.savetxt('./csvs/DRO/reward_diff_with_gamma.csv', Reward_diff, delimiter=',')
    np.savetxt('./csvs/DRO/step_terminate_with_gamma.csv', StepCost, delimiter=',')
    sns.set(rc={'figure.figsize': (16, 12)})
    sns.set_theme(style="whitegrid", font_scale=3)
    dict = {'Gamma Range': base_gamma - 0.02*np.arange(args.gamma_num),
            'Rewards Difference': Reward_diff,
            'Step Cost': StepCost}
    pd = pd.DataFrame(dict)
    sns.lmplot(data=pd, x='Gamma Range', y='Rewards Difference')
    plt.savefig('imgs/DRO/' + 'RewardDiff_Gamma.png', dpi=200)
    wandb.log({"DRO_RewardDiff_Gamma": wandb.Image('imgs/' + 'RewardDiff_Gamma.png')})
    plt.show()
    plt.close()

    sns.lmplot(data=pd, x='Gamma Range', y='Step Cost')
    plt.savefig('imgs/DRO/' + 'StepCost_Gamma.png', dpi=200)
    wandb.log({"DRO_StepCost_Gamma": wandb.Image('imgs/' + 'StepCost_Gamma.png')})
    plt.show()
    plt.close()

    # Exp2: Run DP_EXP

    # 1): fix gamma, adjust buffer size
    base_buffersize = 1000
    DP_EXP_total_rewards = []
    DP_EXP_step_for_terminated = []
    for i in range(args.datasize_num):
        args.buffersize = base_buffersize + i * 500
        exp_name = 'DP_EXP' + '_exp1_' + str(i) + '_' + str(args.buffersize)
        algo = DP_EXP(args)
        DP_EXP_policy, DP_EXP_rewards, step = algo.run()
        DP_EXP_total_rewards.append(DP_EXP_rewards)
        DP_EXP_step_for_terminated.append(step)
        np.savetxt('./learnedpolicy/' + exp_name + '.csv', DP_EXP_policy, delimiter=',')

    Reward_diff = np.array(DP_EXP_total_rewards) - DP_total_rewards
    StepCost = np.array(DP_EXP_step_for_terminated)
    np.savetxt('./csvs/DP_EXP/reward_diff_with_datasize.csv', Reward_diff, delimiter=',')
    np.savetxt('./csvs/DP_EXP/step_terminate_with_datasize.csv', StepCost, delimiter=',')
    # Reward_diff = np.loadtxt('./csvs/DP_EXP/reward_diff_with_datasize.csv', delimiter=',')
    # StepCost = np.loadtxt('./csvs/DP_EXP/step_terminate_with_datasize.csv', delimiter=',')
    sns.set(rc={'figure.figsize': (16, 12)})
    sns.set_theme(style="whitegrid")
    dict = {'Dataset Size': base_buffersize + np.arange(args.datasize_num) * 500,
            'Rewards Difference': np.abs(Reward_diff),
            'Step Cost': StepCost}
    pd = pd.DataFrame(dict)
    sns.lmplot(data=pd, x='Dataset Size', y='Rewards Difference')
    plt.savefig('imgs/DP_EXP/' + 'RewardDiff_Dataset_Size.png', dpi=200)
    wandb.log({"DP_EXP_RewardDiff_Dataset_Size": wandb.Image('imgs/' + 'RewardDiff_Dataset_Size.png')})
    plt.show()
    plt.close()

    sns.lmplot(data=pd, x='Dataset Size', y='Step Cost')
    plt.savefig('imgs/DP_EXP/' + 'StepCost_Dataset_Size.png', dpi=200)
    wandb.log({"DP_EXP_StepCost_Dataset_Size": wandb.Image('imgs/' + 'StepCost_Dataset_Size.png')})
    plt.show()
    plt.close()

    # 2): fix buffer size, adjust gamma
    base_gamma = 1
    DP_EXP_total_rewards = []
    DP_EXP_step_for_terminated = []

    for i in range(args.gamma_num):
        args.gamma = base_gamma - 0.02 * i
        exp_name = 'DP_EXP' + '_exp2_' + str(i) + '_' + str(args.gamma)
        algo = DP_EXP(args)
        DP_EXP_policy, DP_EXP_rewards, step = algo.run()
        DP_EXP_total_rewards.append(DP_EXP_rewards)
        DP_EXP_step_for_terminated.append(step)
        np.savetxt('./learnedpolicy/' + exp_name + '.csv', DP_EXP_policy, delimiter=',')

    Reward_diff = np.array(DP_EXP_total_rewards) - DP_total_rewards
    StepCost = np.array(DP_EXP_step_for_terminated)
    np.savetxt('./csvs/DP_EXP/reward_diff_with_gamma.csv', Reward_diff, delimiter=',')
    np.savetxt('./csvs/DP_EXP/step_terminate_with_gamma.csv', StepCost, delimiter=',')
    sns.set(rc={'figure.figsize': (16, 12)})
    sns.set_theme(style="whitegrid", font_scale=3)
    dict = {'Gamma Range': base_gamma - 0.02 * np.arange(args.gamma_num),
            'Rewards Difference': Reward_diff,
            'Step Cost': StepCost}
    pd = pd.DataFrame(dict)
    sns.lmplot(data=pd, x='Gamma Range', y='Rewards Difference')
    plt.savefig('imgs/DP_EXP/' + 'RewardDiff_Gamma.png', dpi=200)
    wandb.log({"DP_EXP_RewardDiff_Gamma": wandb.Image('imgs/' + 'RewardDiff_Gamma.png')})
    plt.show()
    plt.close()

    sns.lmplot(data=pd, x='Gamma Range', y='Step Cost')
    plt.savefig('imgs/DP_EXP/' + 'StepCost_Gamma.png', dpi=200)
    wandb.log({"DP_EXP_StepCost_Gamma": wandb.Image('imgs/' + 'StepCost_Gamma.png')})
    plt.show()
    plt.close()
