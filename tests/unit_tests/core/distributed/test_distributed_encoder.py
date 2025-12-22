

from unittest import TestCase
from unittest.mock import patch, Mock
import torch
import logging

from unit_tests.test_utils import spawn
tensors_to_send = torch.load("/nvfile-heatstorage/teleai-infra/litian/teletron-refactor/tests/tensor_to_send.pt", weights_only=False)

def producer_process(rank, world_size, q):
    torch.cuda.set_device(rank)
    device = torch.cuda.current_device()
    
    print("producer 000")
    from teletron.core.distributed.base_encoder import BaseEncoder
    print("producer 111")
    torch.distributed.init_process_group(world_size=2, rank=0)
    print("producer 222")
    # args = Mock()
    
    tensors = [tensor.cuda() for tensor in tensors_to_send]
    tensors_info = BaseEncoder._get_tensors_size(tensors, device=device)
    print("producer", tensors_info)
    torch.distributed.send(tensors_info, dst=1)
    packed_tensor = BaseEncoder._pack_tensors(tensors)
    torch.distributed.send(packed_tensor, dst=1)


def consumer_process(rank, world_size, q):
    torch.cuda.set_device(rank)
    device = torch.cuda.current_device()
    
    print("consumer 000")
    from teletron.train.consumer_dataloader import unpack_tensors 
    print("consumer 111")
    torch.distributed.init_process_group(world_size=2, rank=1)
    print("consumer 222")

    tensors_info = torch.empty(10, dtype=torch.int32, device=device)
    torch.distributed.recv(tensors_info, src=0)
    print("consumer", tensors_info)
    tensor_sizes = [(tensors_info[i*2], tensors_info[i*2+1]) for i in range(len(tensors_info)//2)]
    print("consumer", tensor_sizes)
    
    tensor_numels = [size[0] * size[1] for size in tensor_sizes]
    from functools import reduce
    total_size = reduce(lambda x,y: x+y, tensor_numels, 0)
    packed_tensor = torch.empty(total_size, dtype=torch.bfloat16, device=device)
    torch.distributed.recv(packed_tensor, src=0)
    intervals = [0]
    for numel in tensor_numels:
        intervals.append(numel + intervals[-1])
    unpacked_tensors = unpack_tensors(packed_tensor, intervals)
    global tensors_to_send
    tensors = [tensor.cuda() for tensor in tensors_to_send]
    for unpacked_tensor, tensor_size, tensor in zip(unpacked_tensors, tensor_sizes, tensors):
        if not torch.all(unpacked_tensor.view(tensor_size)==tensor):
            print(unpacked_tensor.view(tensor_size)[:5,:5]==tensor[:5,:5])
            print(unpacked_tensor.view(tensor_size))
            print(tensor.view(tensor_size))
            q.put("test fail")
            return 
    q.put('test success')
        
    # sizes = []
    # tensors = [torch.empty(tensor_size) for tensor_size in tensor_sizes]
    
def test_producer_consumer():
    import os 
    os.environ['WORLD_SIZE'] = str(2)
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '12490'

    q = spawn(2, [producer_process, consumer_process])
    msg = q.get()
    assert msg == "test success"
    
def test_pack_unpack(self):
    from teletron.core.distributed.base_encoder import BaseEncoder
    from teletron.train.consumer_dataloader import unpack_tensors 
    tensors = self.tensors_to_send
    packed_tensor = BaseEncoder._pack_tensors(tensors)
    intervals = [10 * 10 * i * (i + 1) // 2 for i in range(6)]
    unpacked_tensors = unpack_tensors(packed_tensor, intervals)
    for tensor, unpacked_tensor in zip(tensors, unpacked_tensors):
        print(tensor.shape, unpacked_tensor.shape)
        self.assertTrue(torch.all(tensor.flatten() == unpacked_tensor))   
    
# if __name__ == "__main__":    
#     import os 
#     os.environ['WORLD_SIZE'] = str(2)
#     os.environ['MASTER_ADDR'] = '127.0.0.1'
#     os.environ['MASTER_PORT'] = '12490'

#     spawn(2, [producer_process, consumer_process])
    # q = spawn(1, consumer_process)

# class TestBaseEncoder(TestCase):
#     def setUp(self):
#         global tensors_to_send
#         self.tensors_to_send = tensors_to_send
    
    # def test_pack_unpack(self):
        # from teletron.core.distributed.base_encoder import BaseEncoder
        # from teletron.train.consumer_dataloader import unpack_tensors 
    #     tensors = self.tensors_to_send
    #     packed_tensor = BaseEncoder._pack_tensors(tensors)
    #     intervals = [10 * 10 * i * (i + 1) // 2 for i in range(6)]
    #     unpacked_tensors = unpack_tensors(packed_tensor, intervals)
    #     for tensor, unpacked_tensor in zip(tensors, unpacked_tensors):
    #         print(tensor.shape, unpacked_tensor.shape)
    #         self.assertTrue(torch.all(tensor.flatten() == unpacked_tensor))
            
    # def test_pack_send_recv(self):
    #     spawn(1, producer_process)
    #     q = spawn(1, consumer_process)
        
    # @patch("teletron.utils.get_args")
    # def producer_process(self, rank, world_size, q, get_args):
    #     torch.distributed.init_process_group(world_size=2, rank=0)
    #     args = Mock()
        
    #     tensors = self.tensors_to_send
    #     tensors_info = BaseEncoder._get_tensors_size(tensors, device=self.device)
    #     print(tensors_info)
    #     torch.distributed.send(tensors_info, dst=1)
        
    
    # @patch("teletron.utils.get_args")
    # def consumer_process(self, rank, world_size, q, get_args):
    #     torch.distributed.init_process_group(world_size=2, rank=1)
    #     args = Mock()
    #     tensors_info = torch.empty(10, dtype=torch.int32)
    #     torch.distributed.recv()
    
