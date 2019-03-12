class config():
    # env config
    render_train = False
    render_test = False
    overwrite_render = True
    record = False
    high = 255.

    # model and training config
    # there are 1702 patients in validation set.
    num_episodes_test = 1702
    grad_clip = True
    clip_val = 20
    saving_freq = 5000
    log_freq = 50
    #default 100
    eval_freq = 100
    soft_epsilon = 0
    use_batch_norm = True
    """
    Enviornment to train on. Can use 'offpol' or 'model'. Otherwise will run on
    test env.
    Off policy env does not take action in step(). Rather, it returns action.
    """
    train_env = 'model'

    # output config
    output_path = "results/final_dqn_" + train_env + "/"
    model_output = output_path + "model.weights/"
    log_path = output_path + "log.txt"
    plot_output = output_path + "scores.png"

    # hyper params
    # 153582 total records for training in mimic
    nsteps_train = 153582
    batch_size = 32
    buffer_size = 500
    target_update_freq = 500
    gamma = 1
    learning_freq = 4
    state_history = 4
    lr_begin = 0.00025
    lr_end = 0.0001
    lr_nsteps = nsteps_train / 2
    eps_begin = 1
    eps_end = 0.01
    eps_nsteps = nsteps_train / 2
    learning_start = 200
