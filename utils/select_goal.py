from sklearn.cluster import k_means
import numpy as np
import multiprocessing


def parrallel_score_samples(kde, samples, thread_count=int(0.875 * multiprocessing.cpu_count())):
    with multiprocessing.Pool(thread_count) as p:
        return np.concatenate(p.map(kde.score_samples, np.array_split(samples, thread_count)))
    
def Select_relabel_goal(batch_data, goals, de, mean_density):
    state = batch_data["observations"]
    goal = batch_data["resampled_goals"]
    batch_size = state.shape[0]
    state_dim = state.shape[-1]
    key_goals_num = goals.shape[0]
    key_goals = goal[:key_goals_num,:]
    key_goals[:,:2] = goals.reshape(key_goals_num,-1)[:,:2]
    index = np.expand_dims(np.arange(batch_size),1)

    state_array = np.repeat(np.expand_dims(state, axis=1), key_goals_num, axis=1)
    key_goals_array = np.repeat(np.expand_dims(key_goals, axis=0), batch_size, axis=0)

    state_goal_array = np.concatenate((state_array.reshape(-1, state_dim)[:,:2], key_goals_array.reshape(-1, state_dim)[:,:2]),axis=-1)
    state_goal_log_density = de.evaluate_log_density(state_goal_array).reshape(batch_size, -1)
    density = np.exp(state_goal_log_density)

    # density_ = density-mean_density/10
    

    for i in range(batch_size):
        d_mean = density[i].mean()

        condition1 = (density[i]>0.8*d_mean)
        condition2 = (density[i]<1.2*d_mean)

        # condition1 = (density[i]>0)
        # condition2 = (density[i]<mean_density)

        idx = np.where((condition1)&(condition2))[0]
        if len(idx)>0:
            g_idx = np.random.choice(idx)
            goal[i] = key_goals_array[i,g_idx]
        else:
            # g_idx = np.random.choice(np.arange(key_goals_num))
            continue


    # if len(density_index) > 0:
    #     idx = np.random.choice(density_index)
    #     selected_goal_index = idx
    # else:
    #     selected_goal_index = density.argsort(axis=1)
    #     selected_goal_index_ = key_goals_num*index + selected_goal_index
    #     selected_goal_index_f = selected_goal_index_.flatten()
    #     s = key_goals_array[selected_goal_index_f].reshape(batch_size, -1, state_dim)
    #     relabel_goals = s[:, -1, :]


    batch_data["resampled_goals"] = goal

    test_state_regoal_array = np.concatenate((state[:,:2], goal[:,:2]),axis=-1)
    test_state_regoal_density = np.clip(np.exp(de.evaluate_log_density(test_state_regoal_array).reshape(batch_size, -1)),0.0,1.0).mean()
    return batch_data, test_state_regoal_density

def Select_relabel_goal2(batch_data, goals, de, policy, mean_density):
    state = batch_data["observations"]
    goal = batch_data["resampled_goals"]
    batch_size = state.shape[0]
    state_dim = state.shape[-1]
    key_goals_num = goals.shape[0]
    key_goals = goal[:key_goals_num,:]
    key_goals[:,:2] = goals.reshape(key_goals_num,-1)[:,:2]
    index = np.expand_dims(np.arange(batch_size),1)

    state_array = np.repeat(np.expand_dims(state, axis=1), key_goals_num, axis=1)
    final_goal_array = np.repeat(np.expand_dims(goal, axis=1), key_goals_num, axis=1)
    key_goals_array = np.repeat(np.expand_dims(key_goals, axis=0), batch_size, axis=0)

    state_goal_array = np.concatenate((final_goal_array.reshape(-1, state_dim)[:,:2], key_goals_array.reshape(-1, state_dim)[:,:2]),axis=-1)


    # multiprocessing
    # state_goal_log_density = parrallel_score_samples(de.kde, state_goal_array)
    state_goal_log_density = parrallel_score_samples(de.kde, state_goal_array)

    # state_goal_log_density = de.evaluate_log_density(state_goal_array).reshape(batch_size, -1)
    density = np.exp(state_goal_log_density)

    # density_ = density-mean_density/10
    # density_ = density
    

    for i in range(batch_size):
        # gg_array = np.concatenate((final_goal_array[i,:,:2], key_goals_array[i,:,:2]), axis=-1)
        # gg_log_density = de.evaluate_log_density(gg_array)
        # density = np.exp(gg_log_density)
        d_mean = density[i].mean()

        condition1 = (density[i]>0.8*d_mean)
        condition2 = (density[i]<1.2*d_mean)

        idx = np.where((condition1)&(condition2))[0]
        if len(idx)>0:
            g_idx = np.random.choice(idx)
        else:
            g_idx = np.random.choice(np.arange(key_goals_num))
        goal[i] = key_goals_array[i,g_idx]

    # if len(density_index) > 0:
    #     idx = np.random.choice(density_index)
    #     selected_goal_index = idx
    # else:
    #     selected_goal_index = density.argsort(axis=1)
    #     selected_goal_index_ = key_goals_num*index + selected_goal_index
    #     selected_goal_index_f = selected_goal_index_.flatten()
    #     s = key_goals_array[selected_goal_index_f].reshape(batch_size, -1, state_dim)
    #     relabel_goals = s[:, -1, :]


    batch_data["resampled_goals"] = goal

    test_state_regoal_array = np.concatenate((state[:,:2], goal[:,:2]),axis=-1)
    test_state_regoal_density = np.clip(np.exp(de.evaluate_log_density(test_state_regoal_array).reshape(batch_size, -1)),0.0,1.0).mean()
    return batch_data, test_state_regoal_density

def Cluster(samples, nums):
    samples_num = samples.shape[0]
    samples_dim = samples.shape[-1]
    data = samples.reshape(samples_num, samples_dim)
    # centers, indices, _ = k_means(data[:,:2],n_clusters=nums,init="k-means++",random_state=0)
    centers = samples[:nums,:]
    return centers