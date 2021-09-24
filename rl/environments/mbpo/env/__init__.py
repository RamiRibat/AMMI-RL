# import gym

# MBPO_ENVIRONMENT_SPECS = (
# 	{
#         'id': 'AntTruncatedObs-v2',
#         'entry_point': (f'fusion.environments.mbpo.env.ant:AntTruncatedObsEnv'),
#     },
# 	{
#         'id': 'HumanoidTruncatedObs-v2',
#         'entry_point': (f'fusion.environments.mbpo.env.humanoid:HumanoidTruncatedObsEnv'),
#     },
# )

# def register_mbpo_environments():
#     for mbpo_environment in MBPO_ENVIRONMENT_SPECS:
#         gym.register(**mbpo_environment)

#     gym_ids = tuple(environment_spec['id'] for environment_spec in  MBPO_ENVIRONMENT_SPECS)

#     return gym_ids

# if __name__ == "__main__":
#     register_mbpo_environments()


from gym.envs.registration import register

register(
    id='AntTruncatedObs-v2',
    entry_point='rl.environments.mbpo.env.ant:AntTruncatedObsEnv',
    max_episode_steps=1000)

register(
    id='HumanoidTruncatedObs-v2',
    entry_point='rl.environments.mbpo.env.humanoid:HumanoidTruncatedObsEnv',
    max_episode_steps=1000)