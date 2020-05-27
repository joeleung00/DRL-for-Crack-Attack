def gen_args(agent, num_episode_per_thread, num_processes, epsilon):
    a = []
    for i in range(num_processes):
        a.append((agent, num_episode_per_thread, i, epsilon))
    return a

def new_td_learning(args):
    set_start_method('spawn')
    agent = DQNAgent(args)
    pre_agent = DQNAgent(args)
    teacher_agent = DQNAgent(args)
    replay_memory = deque(maxlen = MAX_REPLAY_MEMORY_SIZE)
    epsilon = args.initial_epsilon
    epsilon_decay = (args.initial_epsilon - args.final_epsilon) / args.total_iterations
    eval_game(agent, 300)
    
    num_episode_per_thread = args.step_per_iteration // args.NUM_OF_PROCESSES // (GAMEOVER_ROUND)
    outer = tqdm(range(args.total_iterations), desc='Iteration', position=0)
    for epoch in outer:
        thread_args = gen_args(agent, num_episode_per_thread, args.NUM_OF_PROCESSES, epsilon)
        with Pool(processes=args.NUM_OF_PROCESSES) as pool:
            for result in pool.imap(thread_thunk, thread_args):
                replay_memory.extend(result)
        
        if len(replay_memory) >= observation_data:
            pre_agent =  clone_agent(agent, pre_agent)
            data_size = TRAINING_DATA_SIZE if len(replay_memory) >= TRAINING_DATA_SIZE else len(replay_memory)
            agent.train(random.sample(replay_memory, data_size), teacher_agent, agent)
            if epoch > 0:
                teacher_agent = clone_agent(pre_agent, teacher_agent)
    
            eval_game(agent, 500)

            if epsilon > args.final_epsilon:
                epsilon -= epsilon_decay
        

def thread_thunk(args):
    agent = args[0]
    num_episode_per_thread = args[1]
    process_id = args[2]
    epsilon = args[2]
    train_data = []
    if process_id == 0:
        inner = tqdm(range(num_episode_per_thread), desc='Episode', position=1)
    else:
        inner = range(num_episode_per_thread)
    for i in inner:
        train_data.extend(run_episode(agent, epsilon))
    return train_data


def run_episode(agent, epsilon):
    train_data = []
    game = Game(show = False)
    while not game.termination():
        board = copy.deepcopy(game.gameboard.board)
        choice = agent.greedy_policy(board, game.gameboard.get_available_choices(), epsilon)
        _, reward = game.input_pos(choice[0], choice[1])
        next_board = copy.deepcopy(game.gameboard.board) 
        train_data.append((board, choice, reward, next_board))

    return train_data