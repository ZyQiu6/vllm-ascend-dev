# Copyright 2025 Ziyi Qiu
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import ray
import zmq
import math
import msgpack
from enum import Enum

def best_path_node(nodes):
    best_child = None
    best_token = None
    max_reward = -math.inf
    for node in nodes:
        for key in node.children.keys():
            if node.children[key].reward > max_reward:
                best_token = key
                best_child = node.children[key]
    return best_token, best_child

class TrieNode:
    def __init__(self):
        self.children = {}
        self.reward = 0
        
    def best_child(self):
        best_child = None
        best_token = None
        max_reward = -math.inf
        for key in self.children.keys():
            if self.children[key].reward > max_reward:
                best_token = key
                best_child = self.children[key]
        return best_token, best_child
    
    def clear(self):
        self.children.clear()
        self.reward = 0


class CongestionState(Enum):
    SLOW_START = 1
    CONGESTION_AVOIDANCE = 2
    SLOW_INCREASE = 3

class RewardAwareSuffixTree:
    def __init__(self):
        self.root = TrieNode()
        self.subpath_index = {} # use tokens to represent nodes
        self.state = CongestionState.SLOW_START
        self.wnd_size: int = 8 # number of predicted tokens
        self.ssthresh = 16
        self.max_wnd = 28
        self.spec_enable = True
        
    def exist(self, path):
        node = self.root
        for char in path:
            if char not in node.children:
                return False
            node = node.children[char]
        return True
    
    def add_node(self, path, reward):
        """
        :param path: List of token
        :param reward: reward score
        """
        if self.exist(path):
            return
        node = self.root
        for char in path:
            if char not in node.children:
                node.children[char] = TrieNode()
                if char not in self.subpath_index:
                    self.subpath_index[char] = [node.children[char]]
                else:
                    self.subpath_index[char].append(node.children[char])
            node = node.children[char]
            node.reward += reward
    
    def find_path_nodes(self, prefix: list) -> list:
        """
        Fast search path. Prefix may not start from root
        """
        if not prefix:
            return []

        first_element = prefix[0]
        if first_element not in self.subpath_index:
            return []

        start_nodes = self.subpath_index[first_element]
        match_nodes = []
        
        # O(N_starts * L) where N_starts is the number of starting nodes and L is prefix length
        for start_node in start_nodes:
            current_node = start_node
            match_length = 1
            for element in prefix[1:]:
                if element in current_node.children:
                    current_node = current_node.children[element]
                    match_length += 1
                else:
                    break
            if match_length == len(prefix):
                match_nodes.append(current_node)

        return match_nodes
    
    def update_wnd_size(self):
        if self.state == CongestionState.SLOW_START:
            self.wnd_size = min(self.wnd_size * 2, self.ssthresh)
            if self.wnd_size == self.ssthresh:
                self.state = CongestionState.SLOW_INCREASE
        elif self.state == CongestionState.CONGESTION_AVOIDANCE:
            self.wnd_size = max(self.wnd_size // 2, 4)
        elif self.state == CongestionState.SLOW_INCREASE:
            self.wnd_size = min(self.wnd_size + 1, self.max_wnd)

    def predict(self, prefix, accept_length=1):
        if self.state == CongestionState.SLOW_START:
            if accept_length == 1:
                self.state = CongestionState.CONGESTION_AVOIDANCE
            elif accept_length < self.wnd_size:
                self.state = CongestionState.SLOW_INCREASE
        elif self.state == CongestionState.CONGESTION_AVOIDANCE:
            if accept_length >= self.wnd_size:
                self.state = CongestionState.SLOW_INCREASE
        elif self.state == CongestionState.SLOW_INCREASE:
            if accept_length < self.wnd_size:
                self.state = CongestionState.CONGESTION_AVOIDANCE
            else:
                self.state = CongestionState.SLOW_INCREASE
        self.update_wnd_size()

        predicted_tokens = []
        
        matched_nodes = self.find_path_nodes(prefix)
        # print(f"predict matched nodes num: {len(matched_nodes)}")
        next_token, next_node = best_path_node(matched_nodes)
        if next_token:
            predicted_tokens.append(next_token)
            for i in range(self.wnd_size - 1):
                next_token, next_node = next_node.best_child()
                if next_token:
                    predicted_tokens.append(next_token)
                else:
                    break
        
        return predicted_tokens
    
    def clear(self):
        self.root.clear()
        self.subpath_index.clear()

@ray.remote(num_cpus=1)
class SuffixTreeGroup:
    def __init__(self, port: 5555):
        self._dict: dict[str, RewardAwareSuffixTree] = {}
        self.predict_times = 0
        self.effective_times = 0
        self.total_right_length = 0
        self.port = port

    def __len__(self):
        return len(self._dict)
    
    def add_tree(self, prompt_id):
        if prompt_id in self._dict:
            return
        self._dict[prompt_id] = RewardAwareSuffixTree()
    
    def tree_append_node(self, prompt_id, seq, reward):
        if prompt_id not in self._dict:
            raise ValueError(f"{prompt_id} not in GlobalRewardAwareSuffixTreeGroup")
        else:
            self._dict[prompt_id].add_node(seq, reward)
    
    def _predict(self, prompt_id, prefix, accept_length):
        self.predict_times += 1
        if accept_length > 1:
            self.effective_times += 1
        self.total_right_length += (accept_length - 1)
        return self._dict[prompt_id].predict(prefix, accept_length)
    
    def predict(self, prompt_id, prefix, accept_length):
        if (not prompt_id in self._dict) or (not prefix):
            return []
        else:
            return self._predict(prompt_id, prefix, accept_length)
    
    def predict_batch(self, prompt_id_list, prefix_list, accept_length_list):
        import time
        begin_time = time.time()
        result = []
        for i in range(len(prompt_id_list)):
            prompt_id = prompt_id_list[i]
            if (not prompt_id in self._dict) or (not prefix_list[i]):
                result.append([])
            else:
                self.predict_times += 1
                if accept_length_list[i] > 1:
                    self.effective_times += 1
                self.total_right_length += (accept_length_list[i] - 1)
                result.append(self._dict[prompt_id].predict(prefix_list[i], accept_length_list[i]))
        # print(f"Every time predict costs {time.time() - begin_time} s")
        return result
    
    def set(self, key, value):
        self._dict[key] = value
    
    def get(self, key, default=None):
        return self._dict.get(key, default)

    def exist(self, key):
        return key in self._dict

    def get_prompt_ids(self):
        return list(self._dict.keys())
    
    def delete(self, key):
        if key in self._dict:
            del self._dict[key]
            
    def compute_metrics(self):
        return {
            'effective_times': self.effective_times,
            'predict_times': self.predict_times,
            'total_right_length': self.total_right_length,
        }

    def clear(self):
        self._dict.clear()
        self.predict_times = 0
        self.effective_times = 0
        self.total_right_length = 0
    
    def run(self):
        context = zmq.Context()
        socket = context.socket(zmq.REP)
        socket.bind(f"tcp://*:{self.port}")

        self.running = True
        print(f"TreeGroup server started on port {self.port}")
        
        while self.running:
            try:
                message = socket.recv()
                request = msgpack.unpackb(message)
                
                response = self._handle_request(request)
                
                socket.send(msgpack.packb(response))
            except Exception as e:
                error_response = {
                    'status': 'error',
                    'message': str(e)
                }
                socket.send(msgpack.packb(error_response))

    def _handle_request(self, request):
        method = request['method']
        params = request['params']
        
        if method == 'predict':
            return self.predict(
                params['prompt_id'],
                params['prefix'],
                params['accept_length'],
            )
        elif method == 'predict_batch':
            return self.predict_batch(
                params['prompt_id_list'],
                params['prefix_list'],
                params['accept_length_list']
            )
        elif method == 'batch':
            return [self._handle_request(req) for req in params['requests']]
        elif method == 'stop':
            self.running = False
            return True
        else:
            return {'error': f'Unknown method: {method}'}

_num_groups: int = 5 # fixed
history_tree_handle = []
class GlobalRewardAwareSuffixTreeGroup:
    """
    All functions are non-blocking, return futures.
    """
    def __init__(self):
        self.groups = []
        self.prompt_ids = set()
        for i in range(_num_groups):
            try:
                actor_handle = ray.get_actor(f"global_tree_{i}")
                self.groups.append(actor_handle)
            except ValueError:
                print(f"Could not find the global actor.")
        self.server_configs = {}
        for i in range(_num_groups):
            self.server_configs[i] = {
                'host': 'localhost',
                'port': 5555+i,
            }
        self._setup_connections()

    def update_prompt_ids(self):
        prompt_ids = []
        for actor_handle in self.groups:
            prompt_ids.extend(ray.get(actor_handle.get_prompt_ids.remote()))
        self.prompt_ids = set(prompt_ids)

    def __len__(self):
        return len(self.groups)
    
    def _get_partition_id(self, prompt_id: str):
        return (hash(prompt_id) % _num_groups)

    def _get_partition(self, prompt_id: str):
        group_index = hash(prompt_id) % _num_groups
        return self.groups[group_index]
    
    def add_tree(self, prompt_id):
        actor = self._get_partition(prompt_id)
        return actor.add_tree.remote(prompt_id)
    
    def tree_append_node(self, prompt_id, seq, reward):
        actor = self._get_partition(prompt_id)
        return actor.tree_append_node.remote(prompt_id, seq, reward)

    def predict(self, prompt_id, prefix, accept_length):
        actor = self._get_partition(prompt_id)
        return actor.predict.remote(prompt_id, prefix, accept_length)
    
    def predict_batch(self, prompt_id_list, sampled_token_list, prefix_length_list, accept_length_list):
        import time
        begin_time = time.time()
        position_id = []
        params = [{
            'prompt_id_list': [],
            'prefix_list': [],
            'accept_length_list': [],
        } for _ in range(_num_groups)]
        draft_token_list = [None for _ in range(_num_groups)]
        for i in range(len(prompt_id_list)):
            partition_id = self._get_partition_id(prompt_id_list[i])
            params[partition_id]['prompt_id_list'].append(prompt_id_list[i])
            if len(sampled_token_list[i]) < prefix_length_list[i]:
                params[partition_id]['prefix_list'].append([])
            else:
                params[partition_id]['prefix_list'].append(sampled_token_list[i][-prefix_length_list[i]:])
            params[partition_id]['accept_length_list'].append(accept_length_list[i])
            position_id.append(partition_id)
        for i in range(_num_groups):
            draft_token_list[i] = self.groups[i].predict_batch.remote(
                params[i]['prompt_id_list'],
                params[i]['prefix_list'],
                params[i]['accept_length_list']
            )
        for i in range(_num_groups):
            draft_token_list[i] = ray.get(draft_token_list[i])

        iter_list = [iter(draft_tokens) for draft_tokens in draft_token_list]
        result = []
        for i in range(len(prompt_id_list)):
            result.append(next(iter_list[position_id[i]]))
        # print(f"Every time predict costs {time.time() - begin_time} s")
        return result

    def delete(self, prompt_id):
        actor = self._get_partition(prompt_id)
        return actor.delete.remote(prompt_id)

    def exist(self, prompt_id):
        # return (prompt_id in self.prompt_ids)
        actor = self._get_partition(prompt_id)
        return actor.exist.remote(prompt_id)

    def clear(self):
        return [p.clear.remote() for p in self.groups]
    
    def compute_metrics(self):
        tasks = [group.compute_metrics.remote() for group in self.groups]
        metrics_list = ray.get(tasks)
        data = {}
        for key in metrics_list[0].keys():
            data[key] = sum([metrics[key] for metrics in metrics_list])
        effective_percent = data['effective_times'] / data['predict_times'] if data['predict_times'] != 0 \
                                else 0
        effective_length = data['total_right_length'] / data['effective_times'] if data['effective_times'] != 0 \
                                else 0
        res = {
            'speculative_decoding/effective_percent': effective_percent,
            'speculative_decoding/effective_length': effective_length,
        }
        return res

    def _setup_connections(self):
        self.servers = {}
        for server_id, config in self.server_configs.items():
            socket = zmq.Context().socket(zmq.REQ)
            socket.setsockopt(zmq.RCVTIMEO, config.get('timeout', 5000))
            server_url = f"tcp://{config['host']}:{config['port']}"
            socket.connect(server_url)
            self.servers[server_id] = {
                'socket': socket,
                'config': config,
                'url': server_url
            }
        print(f"Connected to {len(self.servers)} servers: {list(self.servers.keys())}")
    
    def post_to_server(self, server_id, request):
        if server_id not in self.servers:
            return {'status': 'error', 'message': f'Unknown server: {server_id}'}
        
        socket = self.servers[server_id]['socket']
        
        try:
            request_json = msgpack.packb(request)
            socket.send(request_json)
            response = socket.recv()
            return msgpack.unpackb(response)
        except Exception as e:
            return RuntimeError({
                'status': 'error',
                'message': f'Communication error: {str(e)}'
            })
        
    def post_to_all(self, request):
        from concurrent.futures import ThreadPoolExecutor, as_completed
        results = {}
        
        with ThreadPoolExecutor(max_workers=len(self.servers)) as executor:
            future_to_server = {
                executor.submit(self.post_to_server, server_id, request): server_id
                for server_id in self.servers
            }
            
            for future in as_completed(future_to_server):
                server_id = future_to_server[future]
                try:
                    results[server_id] = future.result()
                except Exception as e:
                    results[server_id] = {'status': 'error', 'message': str(e)}
        
        return results
        
    def post_broadcast(self, requests):
        results = {}
        for server_id, request in requests.items():
            results[server_id] = self.post_to_server(server_id, request)

        return results
    
    def post_predict_batch(self, prompt_id_list, sampled_token_list, prefix_length_list, accept_length_list):
        import time
        begin_time = time.time()
        position_id = []
        params = {
            server_id: {
                'prompt_id_list': [],
                'prefix_list': [],
                'accept_length_list': [],
            } for server_id in range(_num_groups)}
        draft_token_list = [None for _ in range(_num_groups)]
        for i in range(len(prompt_id_list)):
            partition_id = self._get_partition_id(prompt_id_list[i])
            params[partition_id]['prompt_id_list'].append(prompt_id_list[i])
            if len(sampled_token_list[i]) < prefix_length_list[i]:
                params[partition_id]['prefix_list'].append([])
            else:
                params[partition_id]['prefix_list'].append(sampled_token_list[i][-prefix_length_list[i]:])
            params[partition_id]['accept_length_list'].append(accept_length_list[i])
            position_id.append(partition_id)
        # response = asyncio.run(self.post_broadcast({
        #     i: {'method': 'predict_batch', 'params': params[i]} for i in range(_num_groups)
        # }))
        response = self.post_broadcast({
            i: {'method': 'predict_batch', 'params': params[i]} for i in range(_num_groups)
        })
        draft_token_list = []
        for i in range(_num_groups):
            draft_token_list.append(response[i])

        iter_list = [iter(draft_tokens) for draft_tokens in draft_token_list]
        result = []
        for i in range(len(prompt_id_list)):
            result.append(next(iter_list[position_id[i]]))
        # print(f"Every time predict costs {time.time() - begin_time} s")
        return result
    
    def run_server(self):
        return [p.run.remote() for p in self.groups]
    
    def stop_server(self):
        return self.post_to_all({'method': 'stop', 'params': {}})

def init_history_trees():
    global history_tree_handle
    for i in range(_num_groups):
        actor_handle = SuffixTreeGroup.options(name=f"global_tree_{i}").remote(port=5555+i)
        history_tree_handle.append(actor_handle)
    for i in range(_num_groups):
        try:
            ray.get_actor(f"global_tree_{i}")
            print(f"Actor {i} registered successfully.")
        except ValueError:
            print(f"Actor {i} failed to register.")

def get_history_trees():
    history_trees = GlobalRewardAwareSuffixTreeGroup()
    return history_trees