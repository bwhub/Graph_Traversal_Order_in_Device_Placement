from sim.tf_placement_sim.tf_pl_simulator import ImportantOpsSimulator
import networkx as nx
import pickle
import sys
sys.path.append('sim/')
sys.path.append('../sim/')
sys.path.append('../')
# from important_ops_simulator import ImportantOpsSimulator
# from grouper.group_pruner import neighbor_merge_pruner

# What does PPTFItem stands for?
class PPTFItem(object):

    def __init__(self, pickled_inp_file, n_devs, simplify_tf_reward_model=False,
                 final_size=None, use_new_sim=False, sim_mem_usage=False):
        
        # what does the final_size control?

        # This filter function will never have any effect!
        device_names = ['/device:GPU:%d' % i for i in range(n_devs)]
        # print('device_names = {}'.format(device_names))

        # what is going on here? The filter will return the same list
        gpu_devices = filter(lambda dev: 'GPU' in dev, device_names)
        gpu_devices = list(sorted(gpu_devices))

        # print('gpu_devices = {}'.format(gpu_devices))

        # what is this pickled_inp_file??
        # print('### pickled_inp_file is {}'.format(pickled_inp_file))
        # pickled_inp_file is path to a pickle that stores the computation graph
        with open(pickled_inp_file, 'rb') as f:
            j = pickle.load(f)
            
            # print('### j is of type {}'.format(type(j)))
            # print('### j has the following keys {}'.format(j.keys()))
            
            # j is a Python dictionary with the following keys =>
            #               ['step_stats', 'op_perf', 'ungrouped_mapping', 'optim_mg', 'G', 'argv']
            # - j['ungrouped_mapping'] => Python dictionary that contains the ungroup map
            # - j['G'] => grouped computational graph
            # - j['optim_mg'] => Optimized tensorflow metagraph
            # - j['op_perf'] & j['step_stats'] => 
            #               Measurements captured on two K80 GPU configuration available with p2.8xlarge instances on EC2

            # What's this j['G'] and j['ungrouped_mapping']            
            mg, G, ungroup_map = j['optim_mg'], j['G'], j['ungrouped_mapping']

            # What's the difference between has 'op_perf' or ?not
            if 'op_perf' in j:
                op_perf, step_stats = j['op_perf'], j['step_stats']
                cost_d, out_d, temp_mem, mem_info = [None] * 4
            else:
                cost_d, out_d, temp_mem = j['cost_d'], j['out_d'], j['temp_mem']
                mem_info = j['mem_info']
                op_perf, step_stats = None, None

        if len(nx.get_node_attributes(G, 'mem')) == 0:
            # In case the the nodes do not have 'mem' attribute, call the correct_mem_param funciton.
            print('Correcting mem parameters.')
            # Why on earth is this a function that is not included in the class?
            G = correct_mem_param(G, step_stats, ungroup_map)

        # if final_size is not None:
        #     if len(G) > final_size:
        #       G, ungroup_map = neighbor_merge_pruner(G, ungroup_map, final_size)

        if simplify_tf_reward_model:
            assert False
            d = {}
            for n in G.nodes():
                d[n] = 1
            nx.set_node_attributes(G, d, 'cost')
            nx.set_node_attributes(G, d, 'out_size')
            nx.set_node_attributes(G, d, 'mem')

        # Optimized TensorFlow metagraph
        # tensorflow.core.protobuf.meta_graph_pb2.MetaGraphDef
        self.mg = mg 

        # Python dictionary that contains the ungroup map
        self.ungroup_map = ungroup_map
        # number of devices used in the current placement setting.
        self.n_devs = n_devs
        # list of gpu devices
        self.gpu_devs = gpu_devices
        # whether to use new simulator or LegacySimulator
        self.use_new_sim = use_new_sim
        # grouped computational graph
        self.grouped_G = G
        # wheterh to simulate memory usage
        self.sim_mem_usage = sim_mem_usage
        self.cost_d = cost_d
        self.out_d =  out_d
        # Measurements captured on two K80 GPU configuration available with p2.8xlarge instances on EC2
        self.op_perf = op_perf
        self.step_stats = step_stats

        if self.use_new_sim:
            # Choice of simulator
            self.sim = ImportantOpsSimulator(mg, op_perf, step_stats, device_names)
            # cost_d=cost_d, out_d=out_d, temp_mem=temp_mem,
            # mem_info=mem_info)
        else:
            raise Exception('Using old simulator is locked out by default')
            # from old_simulator import LegacySimulator
            # self.old_sim = LegacySimulator(self.cluster, True)

    """
    This function takes a dictionary from node-center-names to their placements
    To ungroup the placements, we first map each node to its corresponding group index
    and then map grp-index to the name. Finally, this mapped name is looked up in the input
    placement dictionary to find out placement of all the ops.
    """

    def new_simulate(self, p):

        for n in p:
            p[n] = str(p[n])

        ungrouped_pl = self.ungroup_pl(p)

        return self.simulate_full_pl(ungrouped_pl)

    def simulate_full_pl(self, ungrouped_pl):

        if self.sim_mem_usage:

            run_time, start_times, mem_utils = self.sim.simulate(
                ungrouped_pl, sim_mem_usage=True)

            return run_time / 1e6, start_times, mem_utils

        else:

            run_time, start_times = self.sim.simulate(ungrouped_pl)

            run_time /= 1e6  # convert to secs

            return run_time, start_times, None

    def simulate_without_ungroup(self, p):

        for n in p:
            p[n] = str(p[n])

        for op in self.mg.graph_def.node:
            if op.name not in p:
                raise Exception('Without ungrouping needs \
                  specifying full placement')

        return self.simulate_full_pl(p)

    def ungroup_pl(self, pl):
        ungroup_map = self.ungroup_map
        ungrouped_pl = {}

        for op in self.mg.graph_def.node:
            grp_ctr = ungroup_map[op.name]
            ungrouped_pl[op.name] = pl[grp_ctr]

        return ungrouped_pl

    def get_grouped_graph(self):
        return self.grouped_G

    def get_ungroup_map(self):
        return self.ungroup_map

    def simulate(self, *args, **kwargs):
        if self.use_new_sim:
            return self.new_simulate(*args, **kwargs)
        else:
            raise Exception('No longer supported')
            return self.old_simulate(*args, **kwargs)


def ungroup_map_to_adj_l(ungroup_map):
    # Oh boy! Should probably use collection.defaultdict
    d = {}
    for k, v in ungroup_map.items():
        if v in d:
            d[v].append(k)
        else:
            d[v] = [k]
    return d


def correct_mem_param(G, step_stats, ungroup_map):
    print('### Entering correct_mem_param() fucntion')
    out_d_mg = {}
    # What's the difference in step_stats, dev_stats, and node_stats
    for dev_stats in step_stats.dev_stats:
        for node_stats in dev_stats.node_stats:
            node = node_stats.node_name
            for output in node_stats.output:
                allocation = output.tensor_description.allocation_description
                num_bytes = allocation.requested_bytes
                out_d_mg[node] = num_bytes
                # If there is a break function, why the hell use a for loop???
                break

        # Probably will be more efficient if adj do not have to be created multiple times in the for loop
        adj = ungroup_map_to_adj_l(ungroup_map)
        out_d = {}

        for node in G:
            s = 0
            for n in adj[node]:
                s += out_d_mg.get(n, 0)
            out_d[node] = out_d.get(node, 0) + s

        nx.set_node_attributes(G, out_d, 'mem')
        return G
