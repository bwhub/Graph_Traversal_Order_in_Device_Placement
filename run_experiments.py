import time
from datetime import datetime
from benchmark_runner.benchmark_runner import BenchmarkRunner
import pathlib


def run_experiment_one_graph_at_a_time():

    ORDER_LIST = ['alpha', 'topo', 'reversed_topo', 'dfs_preorder', 'dfs_postorder', 'bfs']
    EXEC_PATH = './model/progressive_placer_test.py'
    RESULT_PATH = './placeto_results/test/'
    CONFIG_PATH = './config/config_base.txt'
    REPEAT_TIMES = 10
    PROCESS_COUNT = 20
    DATASET_PATH = './datasets/nmt'

    print('--------------- Experiment Runing ----------------')
    graph_path = pathlib.Path(DATASET_PATH)
    graph_path_list = [x for x in graph_path.iterdir() if x.is_dir()]
    print('There are {} graphs to run.'.format(len(graph_path_list)))
    start_time = time.time()

    for graph_path in graph_path_list:
        run_experiment_on_one_graph(
                                        exec_path=EXEC_PATH,
                                        dataset_path=str(graph_path),
                                        result_path=RESULT_PATH,
                                        config_path=CONFIG_PATH,
                                        repeated_times=REPEAT_TIMES,
                                        order_list=ORDER_LIST,
                                        process_count=PROCESS_COUNT
                                )
    end_time = time.time()
    print('--------------- Experiment finished ----------------')
    print('The total experiment took {} seconds.'.format(end_time - start_time))


def run_experiment_on_one_graph(exec_path, dataset_path, result_path, config_path, repeated_times, order_list, process_count=None):

    runner = BenchmarkRunner(
                                exec_path=exec_path,
                                result_path=result_path+dataset_path.split('/')[-1],
                                base_config_file_path=config_path,
                                repeated_times=repeated_times,
                            )

    runner.add_config('--placement-traversal-order', order_list)
    runner.add_config('-dataset', dataset_path)

    print('--------------- Experiment Runing ----------------')
    print('# {}'.format(datetime.now().strftime('%d/%m/%Y %H:%M:%S')))
    start_time = time.time()

    print('\nExperiments on {} started.'.format(dataset_path))
    if process_count:
        runner.run_test_parallel(process_count=process_count)
    else:
        runner.run_test()
    print('\nExperiments on {} ended.'.format(dataset_path))

    end_time = time.time()
    print('--------------- Experiment finished ----------------')
    print('The experiment took {} seconds.'.format(end_time - start_time))

def run_experiment_max_parallelization():

    ORDER_LIST = ['alpha', 'topo', 'reversed_topo', 'dfs_preorder', 'dfs_postorder', 'bfs']
    EXEC_PATH = './model/progressive_placer_test.py'
    RESULT_PATH = './placeto_results/FOLDER/'
    CONFIG_PATH = './config/config_base.txt'
    REPEAT_TIMES = 10
    PROCESS_COUNT = 1
    DATASET_PATH = './datasets/nmt'

    runner = BenchmarkRunner(
                                exec_path=EXEC_PATH,
                                result_path=RESULT_PATH+DATASET_PATH.split('/')[-1],
                                base_config_file_path=CONFIG_PATH,
                                repeated_times=REPEAT_TIMES,
                            )

    runner.add_config('--placement-traversal-order', ORDER_LIST)
    graph_path = pathlib.Path(DATASET_PATH)
    graph_path_list = [str(x) for x in graph_path.iterdir() if x.is_dir()]
    runner.add_config('-dataset', graph_path_list)

    print('--------------- Experiment Runing ----------------')
    print('# {}'.format(datetime.now().strftime('%d/%m/%Y %H:%M:%S')))
    print('There are {} graphs to run.'.format(len(graph_path_list)))
    start_time = time.time()

    # print('\nExperiments on {} started.'.format(dataset_path))
    if PROCESS_COUNT > 1:
        runner.run_test_parallel(process_count=PROCESS_COUNT)
    else:
        runner.run_test()

    end_time = time.time()
    print('--------------- Experiment finished ----------------')
    print('The total experiment took {} seconds.'.format(end_time - start_time))

if __name__ == "__main__":
    run_experiment_max_parallelization()
    # run_experiment_one_graph_at_a_time()
