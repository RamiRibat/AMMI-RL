import numpy as np

class StaticFns:

    @staticmethod
    def termination_fn(obs, act, next_obs):
        assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 2

        height = next_obs[:, 0]
        angle = next_obs[:, 1]
        not_done =  np.isfinite(next_obs).all(axis=-1) \
                    * np.abs(next_obs[:,1:] < 100).all(axis=-1) \
                    * (height > .7) \
                    * (np.abs(angle) < .2)

        done = ~not_done
        done = done[:,None]

        return done


# class StaticFns:

    @staticmethod
    def reward_fn(obs, act):
        vel_x = obs[:, -6] / 0.02
        power = np.square(act).sum(axis=-1)
        height = obs[:, 0]
        ang = obs[:, 1]
        alive_bonus = 1.0 * (height > 0.7) * (np.abs(ang) <= 0.2)
        rewards = vel_x + alive_bonus - 1e-3*power

        return rewards
