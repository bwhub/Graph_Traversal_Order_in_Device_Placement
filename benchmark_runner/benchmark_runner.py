"""This file contains the class for helping to benchmark Placeto.
"""

import multiprocessing as mp
import subprocess
import time
import itertools as it
import pathlib
import tqdm

# ------------- Remove after finish developing -----------
import pprint
# ------------- Remove after finish developing -----------


class BenchmarkRunner:
    """BenchmarkRunner class for launching device placement experiments."""

    def __init__(self, exec_path, result_path, base_config_file_path, repeated_times, experiment_ind_offset=None):
        """Initializing a benchmark runner instance.
        
        Arguments:
        - exec_path: str path to entry point of experiments.
        - result_path: str path to where to save the result.
        - base_config_file_path: str path to config file of the experiment.
                                    Note that config might be overwritten later.
        - reapeated_times: number of times to repeat the experiments with the same configuration.
        - experiment_ind_offset: offset of experiment index. If None, 0 will be used.

        """
        # path to executable of the experiments
        self.exec_path = exec_path
        # path to results folder
        self.result_path = result_path
        # List of configuration that will be used as base configuration.
        #   Note that base configuration might be overwritten if specified later.
        self.base_config_dict = self._get_config_dict(base_config_file_path)
        # Number of times to repeat the experiment.
        self.repeated_times = repeated_times
        # The start of experiment index. 
        #   Reapeated experiments index will be based on this offset.
        self.experiment_ind_offset = 0
        if experiment_ind_offset:
            self.experiment_ind_offset = experiment_ind_offset
        # Available benchmark options
        self.available_options = self._get_avaiable_benchmark_options()
        # Options for the experiment
        self.experiment_options_dict = self.base_config_dict

        
    def run_single_command(self, cmd, log_path=None):
        """Run a single command and record output if log_path is not None
        
        Arguments:
        - cmd: a single command to run.
        - log_path: path to save the result.
        
        Returns:
        - result: The return value from subprocess.run(), representing a process that has finished.
        """
        start_time = time.time()
        p = subprocess.Popen(cmd.split(), stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        (output, err) = p.communicate()
        end_time = time.time()


        if log_path:
            # mkdir
            pathlib.Path(log_path).mkdir(parents=True, exist_ok=True)
            # log
            with open(log_path+'args.log', 'w') as f:
                f.write(cmd)
                f.write('\nRunning time of the previous command is {} seconds'.format(str(int(end_time - start_time))))
            with open(log_path+'stdout.log', 'w') as f:
                f.write(output.decode('utf-8'))
            with open(log_path+'stderr.log', 'w') as f:
                f.write(err.decode('utf-8'))
        
        return output.decode('utf-8')

    def _get_config_dict(self, file_path):
        """Read and runturn experiment configuration as a dictionary.
        
        Arguments:
        - file_path: str path to config gile of the experiment.
        
        Returns:
        - config_dict: dictionary whose keys represent parameter names 
                            and values represent the actual config.
        
        """
        config_dict = {}

        with open(file_path) as f:
            config = f.readlines()
        
        for line in config:
            config_dict[line.split()[0]] = line.split()[1]

        return config_dict

    def _get_avaiable_benchmark_options(self, exec_path=None):
        """Get avaiable experiment options from the executable."""
        if exec_path == None:
            exec_path = self.exec_path
        result = self.run_single_command('python3 {} --help'.format(exec_path))
        
        options = result.replace('[', '').replace(']', '').split()
        return list(filter(lambda x: '-' in x, options))

    def add_config(self, key, val):
        """Add a single configuration."""
        # assert key in self.available_options, 'Option \"{}\" is not supported'.format(key)
        self.experiment_options_dict[key] = val
    
    def _add_key_2_every_val(self, key, val):
        result = []
        if isinstance(val, list):
            result = [(key + ' ' + v) for v in val]
        else:
            result = [(key + ' ' + val)]
        return result

    def get_testing_list(self):
        name_list = sorted(self.experiment_options_dict)
        # print([self._add_key_2_every_val(name, self.experiment_options_dict[name]) for name in name_list])
        parameter_combination_list = list(it.product(
                                            *((self._add_key_2_every_val(name, self.experiment_options_dict[name]) for name in name_list))
                                                    )
                                        )
        return parameter_combination_list

    def construct_cmd_list(self, test_list):
        """Construct command list ready for benchmark."""
        result = []
        for single_test_list in test_list:
            cmd = 'python3 ' + self.exec_path + ' ' + '  '.join(single_test_list)
            cmd_list = cmd.split()
            order = cmd_list[cmd_list.index('--placement-traversal-order')+1]
            num_dev = cmd_list[cmd_list.index('--n-devs')+1]
            for run_ind in range(self.repeated_times):
                # folder=./results/results_${order}_3dev_run_${run_ind}/
                folder = '{}/results_{}_{}dev_run_{}/'.format(self.result_path, order, num_dev, self.experiment_ind_offset+run_ind)
                # log=results_${order}_3dev_run_${run_ind}_training.log
                log = folder + 'log/'
                cmd_post = ' --id {} --model-folder-prefix {}'.format(self.experiment_ind_offset+run_ind, folder)
                # pprint.pprint((cmd + cmd_post).split())
                result.append(((cmd + cmd_post), log))
        return result
    
    def run_test(self):
        self.test_list = self.get_testing_list()

        cmd_list = self.construct_cmd_list(self.test_list)

        for (cmd, log) in tqdm.tqdm(cmd_list):
            # pprint.pprint('cmd is {}\n log is {}'.format(cmd, log))
            self.run_single_command(cmd, log)
    
    def run_test_parallel(self, process_count):
    # def run_test_parallel(self, log_dir=None, process_count=mp.cpu_count()):
        if process_count > mp.cpu_count():
            print('number of of process should be smaller than cpu_count.')
            process_count = mp.cpu_count()
        
        self.test_list = self.get_testing_list()

        cmd_list = self.construct_cmd_list(self.test_list)
        print('There are {} experiments to run.'.format(len(cmd_list)))
        # print('cmd_list is {}'.format(cmd_list))

        # return
        
        with mp.Pool(process_count) as pool:
            for _ in tqdm.tqdm(pool.starmap_async(self.run_single_command, cmd_list).get()):
                pass

# def main():
#     runner = BenchmarkRunner(
#                                 exec_path='./model/progressive_placer_test.py',
#                                 result_path='./May12_Z220_results',
#                                 base_config_file_path='./config/config_base.txt',
#                                 repeated_times=2,
#                             )

#     runner.add_config('--placement-traversal-order', ['alpha', 'topo', 'reversed_topo', 'dfs_preorder', 'dfs_postorder', 'bfs'])

#     print('--------------- Experiment Runing ----------------')
#     start_time = time.time()
#     # runner.run_test()
#     runner.run_test_parallel(process_count=7)
#     end_time = time.time()
#     print('--------------- Experiment finished ----------------')
#     print('The experiment took {} seconds.'.format(end_time - start_time))

# if __name__ == "__main__":
#     main()