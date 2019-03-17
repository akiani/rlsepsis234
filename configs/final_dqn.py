import datetime
import time


class config():
    # env config
    render_train = False
    render_test = False
    overwrite_render = True
    record = False
    high = 255.

    # model and training config
    # there are 3422 patients in validation set.
    num_episodes_test = 3422
    grad_clip = False
    clip_val = 5
    saving_freq = 5000
    log_freq = 50
    #default 100
    eval_freq = 100
    soft_epsilon = 0
    use_batch_norm = True
    reg_lambda = 5
    reg_thresh = 15
    """
    Enviornment to train on. Can use 'offpol' or 'model'. Otherwise will run on
    test env.
    Off policy env does not take action in step(). Rather, it returns action.
    """
    train_env = 'offpol'

    # output config
    output_path = "results/final_dqn_" + train_env + "/" + datetime.datetime.fromtimestamp(
        time.time()).strftime('%Y%m%dT%H%M%S') + "/"
    model_output = output_path + "model.weights/"
    log_path = output_path + "log.txt"
    plot_output = output_path + "scores.png"

    # hyper params
    # 153581 total records for training in mimic
    nsteps_train = 1000
    batch_size = 32
    buffer_size = 1000
    target_update_freq = 1000
    gamma = 0.99
    learning_freq = 1
    state_history = 4
    lr_begin = 0.0001
    lr_end = 0.0001
    lr_nsteps = nsteps_train / 2
    eps_begin = 1
    eps_end = 0.01
    eps_nsteps = nsteps_train / 2
    learning_start = 200
